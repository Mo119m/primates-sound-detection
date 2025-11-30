"""
Training Module
===============
Complete training pipeline for primate vocalization detection
"""

import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import config
import data_loader
import preprocessing
import augmentation
import model as model_module


def prepare_dataset():
    """
    Load and prepare complete dataset with augmentation
    
    Returns:
        Tuple of (X_train, X_val, y_train, y_val, class_names)
    """
    print("\n" + "=" * 70)
    print("PREPARING DATASET")
    print("=" * 70)
    
    # Step 1: Load raw audio data
    species_data = data_loader.load_species_data()
    background_data = data_loader.load_background_data()
    data_loader.print_data_summary(species_data, background_data)
    
    # Step 2: Convert to mel-spectrograms
    print("\n📊 Converting to Mel-Spectrograms...")
    print("=" * 70)
    
    species_specs = {}
    for species_name, audio_list in species_data.items():
        print(f"\n   Processing {species_name}...")
        specs = []
        for i, (audio, _) in enumerate(audio_list):
            mel_spec = preprocessing.audio_to_melspectrogram(audio)
            specs.append(mel_spec)
            
            if (i + 1) % 50 == 0:
                print(f"   Converted {i + 1}/{len(audio_list)}...")
        
        species_specs[species_name] = specs
        print(f"   ✅ Converted {len(specs)} spectrograms")
    
    # Convert background
    print(f"\n   Processing Background...")
    background_specs = []
    for i, (audio, _) in enumerate(background_data):
        mel_spec = preprocessing.audio_to_melspectrogram(audio)
        background_specs.append(mel_spec)
        
        if (i + 1) % 100 == 0:
            print(f"   Converted {i + 1}/{len(background_data)}...")
    
    print(f"   ✅ Converted {len(background_specs)} background spectrograms")
    
    # Step 3: Augment dataset
    X_aug, y_aug, sample_info = augmentation.augment_dataset(species_specs, background_specs)
    
    # Step 4: Convert spectrograms to RGB images
    print("\n🎨 Converting to RGB Images...")
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
    print(f"   ✅ Created {len(X_images)} RGB images")
    
    # Step 5: Normalize for model input
    X_images = preprocessing.preprocess_for_model(X_images)
    
    # Step 6: Train/validation split
    print("\n📊 Splitting into Train/Validation Sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_images, y_aug,
        test_size=config.VALIDATION_SPLIT,
        random_state=config.RANDOM_SEED,
        stratify=y_aug
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    
    # Print class distribution
    print("\n   Class Distribution:")
    print("   " + "-" * 50)
    for i, class_name in enumerate(config.CLASS_NAMES):
        train_count = np.sum(y_train == i)
        val_count = np.sum(y_val == i)
        total = train_count + val_count
        print(f"   {class_name:30s}: {train_count:5d} train, {val_count:5d} val, {total:5d} total")
    
    print("=" * 70)
    
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
    class_weights = dict(enumerate(weights))
    
    print("\n⚖️  Class Weights:")
    for i, class_name in enumerate(config.CLASS_NAMES):
        print(f"   {class_name:30s}: {class_weights[i]:.4f}")
    
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
    print("\n" + "=" * 70)
    print("TRAINING MODEL")
    print("=" * 70)
    
    # Create model
    model = model_module.create_and_compile_model()
    
    # Calculate class weights if needed
    class_weights = None
    if use_class_weights:
        class_weights = calculate_class_weights(y_train)
    
    # Setup callbacks
    model_save_path = os.path.join(config.MODEL_SAVE_DIR, 'best_model.h5')
    callbacks = model_module.get_callbacks(model_save_path)
    
    print(f"\n🚀 Starting Training...")
    print(f"   Epochs: {config.EPOCHS}")
    print(f"   Batch Size: {config.BATCH_SIZE}")
    print(f"   Model will be saved to: {model_save_path}")
    print("=" * 70)
    
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
    
    print("\n✅ Training completed!")
    
    return model, history


def evaluate_model(model, X_val: np.ndarray, y_val: np.ndarray):
    """
    Evaluate model on validation set
    
    Args:
        model: Trained model
        X_val: Validation images
        y_val: Validation labels
    """
    print("\n" + "=" * 70)
    print("EVALUATING MODEL")
    print("=" * 70)
    
    # Overall evaluation
    results = model.evaluate(X_val, y_val, verbose=0)
    
    print("\n📊 Overall Metrics:")
    for metric_name, value in zip(model.metrics_names, results):
        print(f"   {metric_name}: {value:.4f}")
    
    # Per-class evaluation
    y_pred = model.predict(X_val, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\n📊 Per-Class Performance:")
    print("   " + "-" * 66)
    print(f"   {'Class':<30s} {'Samples':>8s} {'Accuracy':>10s} {'Avg Conf':>10s}")
    print("   " + "-" * 66)
    
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
    
    print("   " + "-" * 66)
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_val, y_pred_classes)
    
    print("\n📊 Confusion Matrix:")
    print("   (rows=true, cols=predicted)")
    print("   " + "-" * 50)
    
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
    
    print("=" * 70)


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
    
    print(f"\n💾 Training history saved to: {save_path}")


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
    print(f"📈 Training plot saved to: {plot_path}")
    
    plt.show()


def run_complete_training_pipeline():
    """
    Run the complete training pipeline from data loading to model evaluation
    
    Returns:
        Trained model
    """
    print("\n" + "=" * 70)
    print("PRIMATE VOCALIZATION DETECTION - TRAINING PIPELINE")
    print("=" * 70)
    
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
    
    print("\n" + "=" * 70)
    print("✅ TRAINING PIPELINE COMPLETED!")
    print("=" * 70)
    
    return model


if __name__ == "__main__":
    # This would be run from the notebook, but can test here
    print("Training Module Ready!")
    print("Call run_complete_training_pipeline() to start training.")
