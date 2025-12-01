"""
Model Module
Define and build the VGG19-based model for primate vocalization classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG19
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import config
import os


def build_model(num_classes: int = config.N_CLASSES,
                input_shape: tuple = (config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS),
                freeze_base: bool = config.FREEZE_BASE_LAYERS) -> keras.Model:
    """
    Build VGG19-based transfer learning model
    
    Args:
        num_classes: Number of output classes
        input_shape: Input image shape
        freeze_base: Whether to freeze VGG19 base layers
    
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
    
    # VGG19 feature extraction
    x = base_model(inputs, training=False)
    
    # Global pooling
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
    
    # Get the base model (VGG19)
    base_model = model.layers[1]  # Assuming VGG19 is the second layer
    
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


def create_and_compile_model() -> keras.Model:
    """
    Convenience function to create and compile model in one step
    
    Returns:
        Compiled model ready for training
    """
    model = build_model()
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
    
    print("\n Model module test completed!")
