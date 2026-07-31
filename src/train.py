"""
Training Module
Complete training pipeline for primate vocalization detection
"""

import numpy as np
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

try:
    from . import config
    from . import data_loader
    from . import preprocessing
    from . import augmentation
    from . import model as model_module
except ImportError:  # Allow running as a standalone script (e.g. in Colab)
    import config
    import data_loader
    import preprocessing
    import augmentation
    import model as model_module


# Window suffixes the pipeline appends when it slices one recording into several
# clips. Stripping them recovers the recording a clip came from, which is the
# unit a validation split has to keep whole.
_WINDOW_SUFFIX_RE = re.compile(
    r"(__t\d+(?:\.\d+)?s(?:__conf[\d.]+)?"      # colger100__t003.0s / field clips
    r"|__\d+s__conf[\d.]+"                      # review clips: __01540s__conf0.980
    r"|_\d+\.\d+s_\d+\.\d+s)$"                  # BirdNET: _807.0s_810.0s
)
# BirdNET writes its own score and a within-file index ahead of the recording
# name (``0.554_2_<recording>_807.0s_810.0s``). Left in place, two segments cut
# from one recording look like two recordings and can be split apart.
_BIRDNET_PREFIX_RE = re.compile(r"^\d+\.\d+_\d+_")
_SAMPLE_INFO_RE = re.compile(r"^(?P<species>.+)_sample(?P<idx>\d+)(?:_aug\d+)?$")


def source_group(path):
    """
    The recording a clip came from.

    Two clips share a group when they are two windows of the same recording, or
    two augmentations of the same window. Either relationship makes them
    near-duplicates, and near-duplicates on opposite sides of a split turn
    validation accuracy into a memorisation score.
    """
    stem = os.path.splitext(os.path.basename(str(path)))[0]
    stem = _BIRDNET_PREFIX_RE.sub("", stem)
    return _WINDOW_SUFFIX_RE.sub("", stem)


def build_source_groups(sample_info, species_paths, background_data):
    """
    A group label for every augmented sample, aligned to ``X_aug``.

    ``augment_dataset`` labels each sample ``<species>_sample<i>_aug<j>`` (or
    ``Background_sample<i>``), where ``i`` indexes the clip list the spectrograms
    were built from -- so the index maps straight back to a file path, and the
    path back to a recording. A sample whose origin cannot be resolved gets a
    unique group of its own, which is the conservative choice: it can never
    place a duplicate on the other side of the split.
    """
    background_paths = [p for _a, p in background_data]
    groups = []
    for k, info in enumerate(sample_info):
        m = _SAMPLE_INFO_RE.match(str(info))
        if not m:
            groups.append(f"__unmatched_{k}")
            continue
        species, idx = m.group("species"), int(m.group("idx"))
        paths = (background_paths if species == "Background"
                 else species_paths.get(species))
        if not paths or idx >= len(paths):
            groups.append(f"__unmatched_{k}")
            continue
        groups.append(f"{species}/{source_group(paths[idx])}")
    return np.asarray(groups)


def grouped_split(X, y, groups, test_size=0.2, seed=42):
    """
    Split so that no recording appears on both sides, keeping classes balanced.

    ``StratifiedGroupKFold`` satisfies both constraints; ``GroupShuffleSplit``
    is the fallback and only satisfies the grouping one. Stratification matters
    here because Colobus_confuser and Colobus_guereza are small classes that a
    careless grouped split can empty out of validation entirely.
    """
    from sklearn.model_selection import GroupShuffleSplit
    n_splits = max(2, int(round(1.0 / max(test_size, 1e-6))))
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                        random_state=seed)
        train_idx, val_idx = next(splitter.split(X, y, groups=groups))
    except Exception:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                     random_state=seed)
        train_idx, val_idx = next(splitter.split(X, y, groups=groups))

    n_groups = len(set(groups))
    shared = set(groups[train_idx]) & set(groups[val_idx])
    print(f"   Grouped by source recording: {n_groups} groups, "
          f"{len(shared)} shared across the split")
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


def prepare_dataset():
    """
    Load and prepare complete dataset with augmentation
    
    Returns:
        Tuple of (X_train, X_val, y_train, y_val, class_names)
    """
    print("PREPARING DATASET")
    
    # Step 1: Load raw audio data. Background is loaded first so its waveforms
    # can serve as the ambient bed for embedding short species clips (removes
    # the zero-padding silence shortcut).
    background_data = data_loader.load_background_data()
    background_pool = [audio for audio, _ in background_data]
    species_data = data_loader.load_species_data(background_pool=background_pool)
    data_loader.print_data_summary(species_data, background_data)
    
    # Step 2: Convert to mel-spectrograms
    print("\n Converting to Mel-Spectrograms...")
    
    species_specs = {}
    # Source file per clip, kept so the train/validation split can group clips
    # that came from the same recording. load_species_data already returns it.
    species_paths = {}
    for species_name, audio_list in species_data.items():
        print(f"\n   Processing {species_name}...")
        specs = []
        for i, (audio, _path) in enumerate(audio_list):
            mel_spec = preprocessing.audio_to_melspectrogram(audio)
            specs.append(mel_spec)

            if (i + 1) % 50 == 0:
                print(f"   Converted {i + 1}/{len(audio_list)}...")

        species_specs[species_name] = specs
        species_paths[species_name] = [p for _a, p in audio_list]
        print(f"  Converted {len(specs)} spectrograms")
    
    # Convert background
    print(f"\n   Processing Background")
    background_specs = []
    for i, (audio, _) in enumerate(background_data):
        mel_spec = preprocessing.audio_to_melspectrogram(audio)
        background_specs.append(mel_spec)
        
        if (i + 1) % 100 == 0:
            print(f"   Converted {i + 1}/{len(background_data)}...")
    
    print(f"  Converted {len(background_specs)} background spectrograms")
    
    # Step 3: Augment dataset
    X_aug, y_aug, sample_info = augmentation.augment_dataset(species_specs, background_specs)
    
    # Step 4: Convert spectrograms to RGB images
    print("\n Converting to RGB Images")
    X_images = []
    for i, spec in enumerate(X_aug):
        # Normalize and resize
        spec_norm = preprocessing.normalize_spectrogram(spec)
        spec_resized = preprocessing.resize_spectrogram(spec_norm)
        rgb_image = preprocessing.spectrogram_to_rgb(spec_resized)
        X_images.append(rgb_image)
        
        if (i + 1) % 500 == 0:
            print(f"   Processed {i + 1}/{len(X_aug)}...")
    
    X_images = np.array(X_images)
    print(f"  Created {len(X_images)} RGB images")
    
    # Step 5: Normalize for model input
    X_images = preprocessing.preprocess_for_model(X_images)
    
    # Step 6: Train/validation split, grouped by source recording.
    #
    # A random split here is not a validation set. Every clip becomes
    # AUGMENTATION_MULTIPLIER (7, or 9 for Colobus) augmented images, so with a
    # 20 % random split the chance that all of one clip's variants land in
    # training is 0.8**7 = 0.21 -- about 79 % of clips put a near-duplicate of a
    # validation image into training. The Colobus class compounds it: its 617
    # windows are cut from 172 source recordings at a 1 s hop, so adjacent
    # windows share half their audio before augmentation even starts.
    #
    # That is the gap between 98.12 % validation accuracy and 41.0 % field
    # precision. Grouping on the source recording closes both leaks at once:
    # every image derived from one recording, however augmented or windowed,
    # falls on the same side of the split.
    print("\n Splitting into Train/Validation Sets")
    groups = build_source_groups(sample_info, species_paths, background_data)
    X_train, X_val, y_train, y_val = grouped_split(
        X_images, y_aug, groups,
        test_size=config.VALIDATION_SPLIT,
        seed=config.RANDOM_SEED,
    )

    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    
    # Print class distribution
    print("\n   Class Distribution:")
    for i, class_name in enumerate(config.CLASS_NAMES):
        train_count = np.sum(y_train == i)
        val_count = np.sum(y_val == i)
        total = train_count + val_count
        print(f"   {class_name:30s}: {train_count:5d} train, {val_count:5d} val, {total:5d} total")
    
    
    return X_train, X_val, y_train, y_val, config.CLASS_NAMES


def calculate_class_weights(y_train: np.ndarray) -> dict:
    """
    Calculate class weights to handle imbalance
    
    Args:
        y_train: Training labels
    
    Returns:
        Dictionary of class weights
    """
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes.tolist(), weights))

    print("\n Class Weights:")
    for i, class_name in enumerate(config.CLASS_NAMES):
        if i in class_weights:
            print(f"   {class_name:30s}: {class_weights[i]:.4f}")
        else:
            print(f"   {class_name:30s}: MISSING — 0 training samples!")

    return class_weights


def train_model(X_train: np.ndarray, 
               X_val: np.ndarray,
               y_train: np.ndarray,
               y_val: np.ndarray,
               use_class_weights: bool = True) -> tuple:
    """
    Train the model
    
    Args:
        X_train: Training images
        X_val: Validation images
        y_train: Training labels
        y_val: Validation labels
        use_class_weights: Whether to use class weights
    
    Returns:
        Tuple of (trained_model, history)
    """

    print("TRAINING MODEL")
    
    # Create model
    model = model_module.create_and_compile_model()
    
    # Calculate class weights if needed
    class_weights = None
    if use_class_weights:
        class_weights = calculate_class_weights(y_train)
    
    # Setup callbacks
    model_save_path = os.path.join(config.MODEL_SAVE_DIR, 'best_model.h5')
    callbacks = model_module.get_callbacks(model_save_path)
    
    print(f"\n  Starting Training...")
    print(f"   Epochs: {config.EPOCHS}")
    print(f"   Batch Size: {config.BATCH_SIZE}")
    print(f"   Model will be saved to: {model_save_path}")

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n Training completed")
    
    return model, history


def evaluate_model(model, X_val: np.ndarray, y_val: np.ndarray):
    """
    Evaluate model on validation set
    
    Args:
        model: Trained model
        X_val: Validation images
        y_val: Validation labels
    """
    print("EVALUATING MODEL")
    
    # Overall evaluation
    results = model.evaluate(X_val, y_val, verbose=0)
    
    print("\n Overall Metrics:")
    for metric_name, value in zip(model.metrics_names, results):
        print(f"   {metric_name}: {value:.4f}")
    
    # Per-class evaluation
    y_pred = model.predict(X_val, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\n Per-Class Performance:")
    print(f"   {'Class':<30s} {'Samples':>8s} {'Accuracy':>10s} {'Avg Conf':>10s}")
    
    for i, class_name in enumerate(config.CLASS_NAMES):
        class_mask = y_val == i
        class_samples = np.sum(class_mask)
        
        if class_samples > 0:
            class_correct = np.sum((y_pred_classes == i) & class_mask)
            class_accuracy = class_correct / class_samples
            
            # Average confidence for this class
            class_confidences = y_pred[class_mask, i]
            avg_confidence = np.mean(class_confidences)
            
            print(f"   {class_name:<30s} {class_samples:>8d} {class_accuracy:>10.2%} {avg_confidence:>10.4f}")
    
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_val, y_pred_classes)

    
    # Header
    header = "   " + " " * 15
    for class_name in config.CLASS_NAMES:
        header += f"{class_name[:8]:>10s}"
    print(header)
    
    # Rows
    for i, class_name in enumerate(config.CLASS_NAMES):
        row = f"   {class_name:<15s}"
        for j in range(len(config.CLASS_NAMES)):
            row += f"{cm[i, j]:>10d}"
        print(row)
    


def save_training_history(history, save_path: str):
    """
    Save training history to file
    
    Args:
        history: Training history object
        save_path: Path to save history
    """
    import json
    
    # Convert history to serializable format
    history_dict = {}
    for key, values in history.history.items():
        history_dict[key] = [float(v) for v in values]
    
    with open(save_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print(f"\n Training history saved to: {save_path}")


def plot_training_history(history):
    """
    Plot training history
    
    Args:
        history: Training history object
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(config.MODEL_SAVE_DIR, 'training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f" Training plot saved to: {plot_path}")
    
    plt.show()


def run_complete_training_pipeline():
    """
    Run the complete training pipeline from data loading to model evaluation
    
    Returns:
        Trained model
    """
    print("PRIMATE VOCALIZATION DETECTION - TRAINING PIPELINE")
    
    # Print configuration
    config.print_config_summary()
    
    # Prepare dataset
    X_train, X_val, y_train, y_val, class_names = prepare_dataset()
    
    # Train model
    model, history = train_model(X_train, X_val, y_train, y_val)
    
    # Evaluate model
    evaluate_model(model, X_val, y_val)
    
    # Save training history
    history_path = os.path.join(config.MODEL_SAVE_DIR, 'training_history.json')
    save_training_history(history, history_path)
    
    # Plot training history
    plot_training_history(history)
    
    print(" TRAINING PIPELINE COMPLETED!")
    
    return model


if __name__ == "__main__":
    # This would be run from the notebook, but can test here
    print("Training Module Ready")
    print("Call run_complete_training_pipeline() to start training.")
