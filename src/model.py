"""
Model Module
Define and build the VGG19-based model for primate vocalization classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG19
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os

try:
    from . import config
except ImportError:  # Allow running as a standalone script (e.g. in Colab)
    import config


def build_model(num_classes: int = config.N_CLASSES,
                input_shape: tuple = (config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS),
                freeze_base: bool = config.FREEZE_BASE_LAYERS,
                pooling: str = 'gap') -> keras.Model:
    """
    Build VGG19-based transfer learning model

    Args:
        num_classes: Number of output classes
        input_shape: Input image shape
        freeze_base: Whether to freeze VGG19 base layers
        pooling: How to pool the VGG19 feature map before the dense head.
            - 'gap' (default): a single GlobalAveragePooling2D over the whole
              feature map. Simple, but averages away *where* in frequency the
              energy sits, so a low roar and a high rhythmic bird trill collapse
              to similar feature vectors.
            - 'freq_bands': split the feature map along the frequency axis into
              low/mid/high bands and pool each separately, then concatenate. The
              dense head can then tell a low-frequency Colobus roar from a
              high-frequency bird call. Targets the Colobus vs bird confusion.
            - 'temporal': pool away the frequency axis but KEEP time, then model
              the time sequence with 1D convolutions. Preserves WHEN energy
              occurs (V7 head).
            - 'temporal_freq': like 'temporal' but splits frequency into 4
              bands before temporal modelling. Each band gets its own Conv1D
              stream, then all bands merge for a cross-band Conv1D + BiLSTM.
              Preserves both WHEN and WHERE (low/mid/high) energy occurs (V8).

    Returns:
        Compiled Keras model
    """
    print("\n Building Model")

    # Load VGG19 with pretrained ImageNet weights
    base_model = VGG19(
        weights=config.PRETRAINED_WEIGHTS,
        include_top=False,
        input_shape=input_shape
    )

    # Freeze base model layers if specified
    if freeze_base:
        base_model.trainable = False
        print("   Frozen VGG19 base layers")
    else:
        print("   VGG19 base layers trainable")

    # Build the model
    inputs = keras.Input(shape=input_shape)

    # VGG19 feature extraction. The 'temporal' head taps an intermediate block
    # instead of the full base output, so it builds its own feature extractor
    # below; running the full base here too would leave a dangling second VGG.
    if pooling not in ('temporal', 'temporal_freq'):
        x = base_model(inputs, training=False)

    # Pooling. The feature map is (batch, freq, time, channels): the height axis
    # is frequency (row 0 ~ FMIN, last row ~ FMAX) because the mel-spectrogram is
    # (n_mels, time) before the square resize.
    if pooling == 'freq_bands':
        n_freq = x.shape[1]
        c1 = max(1, n_freq // 3)
        c2 = n_freq - (n_freq // 3)
        # Cropping2D keeps a contiguous frequency band by cropping the rows above
        # and below it (axis order ((top, bottom), (left, right))). Standard
        # serializable layers, so load_model needs no custom objects/safe_mode.
        low = layers.Cropping2D(((0, n_freq - c1), (0, 0)), name='freq_low')(x)
        mid = layers.Cropping2D(((c1, n_freq - c2), (0, 0)), name='freq_mid')(x)
        high = layers.Cropping2D(((c2, 0), (0, 0)), name='freq_high')(x)
        x = layers.Concatenate(name='freq_band_pool')([
            layers.GlobalAveragePooling2D()(low),
            layers.GlobalAveragePooling2D()(mid),
            layers.GlobalAveragePooling2D()(high),
        ])
        print(f"   Frequency-band pooling: bands [0:{c1}, {c1}:{c2}, {c2}:{n_freq}]")
    elif pooling in ('temporal', 'temporal_freq'):
        # CRNN head -- tells a real call (discrete bursts with rhythm) from an
        # insect pulse train or continuous noise, a distinction that only exists
        # in TIME and that 'gap'/'freq_bands' average away.
        #
        # 'temporal': pools away frequency entirely (V7).
        # 'temporal_freq': splits frequency into N_FREQ_BANDS bands BEFORE
        #   temporal modelling so the head can also learn WHERE in frequency
        #   energy sits (Colobus roar ~512Hz vs Cernic hack ~1kHz vs bird >2kHz).
        #   Each band gets its own Conv1D+BiLSTM stream, concatenated at the end.
        tap = 'block4_conv4'
        feat = keras.Model(base_model.input,
                           base_model.get_layer(tap).output,
                           name='vgg19_temporal')  # 'vgg' prefix so
                                                    # unfreeze_base_model finds it
        fmap = feat(inputs, training=False)         # (b, freq, time, channels)
        n_freq = fmap.shape[1]
        n_time = fmap.shape[2]
        n_ch = fmap.shape[3]

        if pooling == 'temporal':
            # V7 path: average over frequency, keep time only.
            x = layers.AveragePooling2D(pool_size=(n_freq, 1), name='freq_pool')(fmap)
            x = layers.Reshape((n_time, n_ch), name='time_sequence')(x)
            x = layers.Conv1D(256, 3, padding='same', name='temporal_conv1')(x)
            x = layers.BatchNormalization(name='temporal_bn1')(x)
            x = layers.Activation('relu', name='temporal_relu1')(x)
            x = layers.Conv1D(256, 3, padding='same', name='temporal_conv2')(x)
            x = layers.BatchNormalization(name='temporal_bn2')(x)
            x = layers.Activation('relu', name='temporal_relu2')(x)
            x = layers.Bidirectional(
                layers.LSTM(128, return_sequences=True, dropout=0.3),
                name='temporal_bilstm')(x)
            x = layers.Concatenate(name='temporal_pool')([
                layers.GlobalMaxPooling1D()(x),
                layers.GlobalAveragePooling1D()(x),
            ])
            print(f"   Temporal CRNN head: tap {tap} -> freq-pooled to "
                  f"({n_time} steps x {n_ch} ch) -> Conv1D x2 -> BiLSTM")
        else:
            # temporal_freq: split frequency into bands, each gets its own
            # temporal pathway, then merge.  The model explicitly sees WHICH
            # frequency band is active at each time step.
            N_BANDS = 4
            band_size = n_freq // N_BANDS
            band_outputs = []
            band_boundaries = []
            for b in range(N_BANDS):
                f_start = b * band_size
                f_end = n_freq if b == N_BANDS - 1 else (b + 1) * band_size
                band_boundaries.append((f_start, f_end))
                bh = f_start
                crop_top = bh
                crop_bot = n_freq - f_end
                band = layers.Cropping2D(
                    ((crop_top, crop_bot), (0, 0)),
                    name=f'freq_band_{b}')(fmap)
                band_h = f_end - f_start
                band = layers.AveragePooling2D(
                    pool_size=(band_h, 1),
                    name=f'band_{b}_freq_pool')(band)
                band = layers.Reshape(
                    (n_time, n_ch),
                    name=f'band_{b}_seq')(band)
                band = layers.Conv1D(
                    128, 3, padding='same',
                    name=f'band_{b}_conv1')(band)
                band = layers.BatchNormalization(
                    name=f'band_{b}_bn1')(band)
                band = layers.Activation(
                    'relu', name=f'band_{b}_relu1')(band)
                band_outputs.append(band)

            # Stack bands: (batch, time, N_BANDS * 128)
            x = layers.Concatenate(
                axis=-1, name='band_merge')(band_outputs)
            # Cross-band temporal convolution
            x = layers.Conv1D(256, 3, padding='same',
                              name='cross_band_conv')(x)
            x = layers.BatchNormalization(name='cross_band_bn')(x)
            x = layers.Activation('relu', name='cross_band_relu')(x)
            # Recurrent layer sees all bands together over time
            x = layers.Bidirectional(
                layers.LSTM(128, return_sequences=True, dropout=0.3),
                name='temporal_bilstm')(x)
            x = layers.Concatenate(name='temporal_pool')([
                layers.GlobalMaxPooling1D()(x),
                layers.GlobalAveragePooling1D()(x),
            ])
            print(f"   Temporal-freq CRNN head: tap {tap} -> "
                  f"{N_BANDS} freq bands x {n_time} time steps -> "
                  f"per-band Conv1D -> merge -> cross-band Conv1D -> BiLSTM")
    else:
        x = layers.GlobalAveragePooling2D()(x)

    # Dense layers
    x = layers.Dense(512, activation='relu', name='dense_512')(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)
    
    x = layers.Dense(256, activation='relu', name='dense_256')(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name='primate_vocalization_model')
    
    print(f"\n Model Architecture:")
    print(f"   Input Shape: {input_shape}")
    print(f"   Base Model: VGG19 ({len(base_model.layers)} layers)")
    print(f"   Output Classes: {num_classes}")
    print(f"   Total Parameters: {model.count_params():,}")
    
    # Count trainable parameters
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"   Trainable Parameters: {trainable_params:,}")
    
    return model


def compile_model(model: keras.Model,
                 learning_rate: float = config.LEARNING_RATE) -> keras.Model:
    """
    Compile model with optimizer, loss, and metrics
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer
    
    Returns:
        Compiled model
    """
    print("\n Compiling Model")
    
    # Optimizer
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Loss function
    loss = keras.losses.SparseCategoricalCrossentropy()
    
    # Metrics
    metrics = [
        keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
        keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top_2_accuracy')
    ]
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    print("   Model compiled!")
    print(f"   Optimizer: Adam (lr={learning_rate})")
    print(f"   Loss: Sparse Categorical Crossentropy")
    print(f"   Metrics: Accuracy, Top-2 Accuracy")
    
    return model


def get_callbacks(model_save_path: str) -> list:
    """
    Create training callbacks
    
    Args:
        model_save_path: Path to save best model
    
    Returns:
        List of callbacks
    """
    callbacks = []
    
    # Early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=config.PATIENCE,
        min_delta=config.MIN_DELTA,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stop)
    
    # Model checkpoint
    checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # Reduce learning rate on plateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    return callbacks


def unfreeze_base_model(model: keras.Model, 
                       num_blocks_to_unfreeze: int = config.UNFREEZE_LAST_N_BLOCKS) -> keras.Model:
    """
    Unfreeze last N blocks of VGG19 for fine-tuning
    
    Args:
        model: Compiled model
        num_blocks_to_unfreeze: Number of VGG19 blocks to unfreeze from the end
    
    Returns:
        Model with unfrozen layers
    """
    print(f"\n Unfreezing last {num_blocks_to_unfreeze} block(s) of VGG19...")

    # Find the VGG19 base model by name rather than positional index so the
    # function keeps working if a preprocessing layer is added before it.
    base_model = None
    for layer in model.layers:
        if isinstance(layer, keras.Model) and layer.name.lower().startswith('vgg'):
            base_model = layer
            break
    if base_model is None:
        # Fall back to the previous behaviour but emit a clear warning rather
        # than silently grabbing the wrong layer.
        print("   Warning: could not locate a nested VGG model by name; "
              "falling back to model.layers[1]")
        base_model = model.layers[1]

    # VGG19 has 5 blocks, each block ends with a MaxPooling layer
    block_layer_names = []
    for layer in base_model.layers:
        if 'pool' in layer.name:
            block_layer_names.append(layer.name)
    
    # Determine which layers to unfreeze
    if num_blocks_to_unfreeze > 0 and num_blocks_to_unfreeze <= len(block_layer_names):
        unfreeze_from = block_layer_names[-num_blocks_to_unfreeze]
        
        # Unfreeze layers
        set_trainable = False
        for layer in base_model.layers:
            if layer.name == unfreeze_from:
                set_trainable = True
            layer.trainable = set_trainable
        
        # Count trainable parameters
        trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
        print(f"   Unfrozen from layer: {unfreeze_from}")
        print(f"   Trainable Parameters: {trainable_params:,}")
    else:
        print(f"   Invalid num_blocks_to_unfreeze: {num_blocks_to_unfreeze}")
    
    return model


def create_and_compile_model(pooling: str = None) -> keras.Model:
    """
    Convenience function to create and compile model in one step

    Args:
        pooling: pooling head to use ('gap' | 'freq_bands' | 'temporal' |
            'temporal_freq').
            Defaults to config.MODEL_POOLING so the standard training pipeline
            can switch heads via the PRIMATE_MODEL_POOLING env var without
            editing code.

    Returns:
        Compiled model ready for training
    """
    if pooling is None:
        pooling = getattr(config, 'MODEL_POOLING', 'gap')
    model = build_model(pooling=pooling)
    model = compile_model(model)
    return model


def load_trained_model(model_path: str) -> keras.Model:
    """
    Load a trained model from file
    
    Args:
        model_path: Path to saved model
    
    Returns:
        Loaded Keras model
    """
    print(f"\n Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = keras.models.load_model(model_path)
    print("   Model loaded successfully!")
    
    return model


def print_model_summary(model: keras.Model):
    """
    Print detailed model summary
    
    Args:
        model: Keras model
    """

    print("MODEL SUMMARY")

    model.summary()



if __name__ == "__main__":
    # Test model building
    print("Testing Model Module...")
    config.print_config_summary()
    
    # Build model
    model = build_model()
    print_model_summary(model)
    
    # Compile model
    model = compile_model(model)
    
    # Test model with random input
    print("\n Testing model with random input")
    test_input = tf.random.normal((1, config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS))
    test_output = model(test_input, training=False)
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {test_output.shape}")
    print(f"   Output probabilities: {test_output.numpy()}")
    
    print("\n Model module test completed")
