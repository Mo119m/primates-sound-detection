"""
Configuration file for Primate Vocalization Detection Pipeline
All paths, parameters, and settings are defined here for easy modification.
When adding new species or updating data, only need to modify this file.
"""

import os

# GOOGLE DRIVE PATHS (Modify these based on Drive structure)
DRIVE_ROOT = "/content/drive/MyDrive/chimp-audio"
AUDIO_ROOT = os.path.join(DRIVE_ROOT, "audio")
LONG_AUDIO_ROOT = os.path.join(DRIVE_ROOT, "long_audio")

# SPECIES CONFIGURATION
# Add or remove species here 
SPECIES_FOLDERS = {
    'Cercocebus_torquatus': 'Cercocebus torquatus hack 5s',
    'Colobus_guereza': 'Colobus guereza Clips 5s',
    # 'Cercopithecus_nictitans': 'Cercopithecus nictitans hack 5s',  # Uncomment when more data available
}

# Background noise folders (will be combined into single "Background" class)
BACKGROUND_FOLDERS = [
    'background noise Clips 5sec',
    'wrong classified'
]

# AUDIO PARAMETERS
SAMPLE_RATE = 44100  # Hz
CLIP_DURATION = 5.0  # seconds
WINDOW_SIZE = 5.0  # seconds (for detection sliding window)
WINDOW_STRIDE = 2.5  # seconds (50% overlap)

# MEL-SPECTROGRAM PARAMETERS
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
FMIN = 20  # Hz (minimum frequency)
FMAX = 8000  # Hz (maximum frequency, adjust based on primate vocalizations)

# Target image size for VGG19 (will resize spectrogram to this)
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3

# DATA AUGMENTATION PARAMETERS 
AUGMENTATION_CONFIG = {
    'original': 1,  # Keep 1 original version
    'background_noise_mix': 3,  # Mix with 3 different background samples
    'time_chop': 1,  # 1 time cropping augmentation
    'freq_chop': 1,  # 1 frequency cropping augmentation
    'translate': 1,  # 1 frequency translation augmentation
}

# Background noise mixing parameters
BG_MIX_SNR_RANGE = (-5, 10)  # SNR in dB (signal-to-noise ratio range)

# Geometric augmentation parameters
CHOP_RANGE = (0.1, 0.3)  # Crop 10-30% from edges
TRANSLATE_RANGE = (-20, 20)  # Frequency bins to shift

# TRAIN/VALIDATION SPLIT
VALIDATION_SPLIT = 0.2  # 20% for validation
RANDOM_SEED = 42

# MODEL PARAMETERS
MODEL_NAME = 'VGG19'
PRETRAINED_WEIGHTS = 'imagenet'
FREEZE_BASE_LAYERS = True  # Freeze VGG19 base layers initially
UNFREEZE_LAST_N_BLOCKS = 1  # Fine-tune last N blocks later (optional)

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001
DROPOUT_RATE = 0.5

# Early stopping
PATIENCE = 10  # Stop if validation loss doesn't improve for N epochs
MIN_DELTA = 0.001  # Minimum change to qualify as improvement

# DETECTION PARAMETERS
DETECTION_CONFIDENCE_THRESHOLD = 0.7  # Only keep detections above this
NMS_IOU_THRESHOLD = 0.5  # Non-maximum suppression overlap threshold

# OUTPUT PATHS
OUTPUT_ROOT = os.path.join(DRIVE_ROOT, "outputs")
PROCESSED_DATA_DIR = os.path.join(OUTPUT_ROOT, "processed_data")
MODEL_SAVE_DIR = os.path.join(OUTPUT_ROOT, "models")
DETECTION_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "detections")
VISUALIZATION_DIR = os.path.join(OUTPUT_ROOT, "visualizations")

# Create output directories if they don't exist
for directory in [OUTPUT_ROOT, PROCESSED_DATA_DIR, MODEL_SAVE_DIR, 
                  DETECTION_OUTPUT_DIR, VISUALIZATION_DIR]:
    os.makedirs(directory, exist_ok=True)


# DERIVED PARAMETERS
N_CLASSES = len(SPECIES_FOLDERS) + 1  # +1 for Background class
CLASS_NAMES = list(SPECIES_FOLDERS.keys()) + ['Background']

# Calculate expected number of samples per species after augmentation
AUGMENTATION_MULTIPLIER = sum(AUGMENTATION_CONFIG.values())

# HELPER FUNCTIONS
def print_config_summary():
    """Print a summary of the current configuration"""
    print("PRIMATE VOCALIZATION DETECTION - CONFIGURATION SUMMARY")
    print(f"\n Data Paths:")
    print(f"   Audio Root: {AUDIO_ROOT}")
    print(f"   Long Audio Root: {LONG_AUDIO_ROOT}")
    print(f"   Output Root: {OUTPUT_ROOT}")
    
    print(f"\n Species to Detect ({len(SPECIES_FOLDERS)}):")
    for i, (key, folder) in enumerate(SPECIES_FOLDERS.items(), 1):
        print(f"   {i}. {key} <- {folder}")
    
    print(f"\n Background Sources ({len(BACKGROUND_FOLDERS)}):")
    for i, folder in enumerate(BACKGROUND_FOLDERS, 1):
        print(f"   {i}. {folder}")
    
    print(f"\n Audio Parameters:")
    print(f"   Sample Rate: {SAMPLE_RATE} Hz")
    print(f"   Clip Duration: {CLIP_DURATION}s")
    print(f"   Window Size/Stride: {WINDOW_SIZE}s / {WINDOW_STRIDE}s")
    
    print(f"\n Mel-Spectrogram:")
    print(f"   N_FFT: {N_FFT}, Hop: {HOP_LENGTH}")
    print(f"   Mel Bins: {N_MELS}, Freq Range: {FMIN}-{FMAX} Hz")
    print(f"   Target Image Size: {IMG_HEIGHT}x{IMG_WIDTH}x{IMG_CHANNELS}")
    
    print(f"\n Data Augmentation (Multiplier: {AUGMENTATION_MULTIPLIER}x):")
    for aug_type, count in AUGMENTATION_CONFIG.items():
        print(f"   {aug_type}: {count}")
    
    print(f"\n Model:")
    print(f"   Architecture: {MODEL_NAME}")
    print(f"   Classes: {N_CLASSES} ({', '.join(CLASS_NAMES)})")
    print(f"   Batch Size: {BATCH_SIZE}, Epochs: {EPOCHS}")
    print(f"   Learning Rate: {LEARNING_RATE}, Dropout: {DROPOUT_RATE}")
    
    print(f"\n Detection:")
    print(f"   Confidence Threshold: {DETECTION_CONFIDENCE_THRESHOLD}")
    print(f"   NMS IOU Threshold: {NMS_IOU_THRESHOLD}")
    

if __name__ == "__main__":
    print_config_summary()
