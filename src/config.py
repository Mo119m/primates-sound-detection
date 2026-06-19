"""
Configuration file for Primate Vocalization Detection Pipeline
All paths, parameters, and settings are defined here for easy modification.
When adding new species or updating data, only need to modify this file.
"""

import os

# DATA ROOT PATH
# Defaults to the Google Drive layout used in the Colab notebooks, but can be
# overridden via the PRIMATE_DATA_ROOT environment variable so the pipeline can
# run locally or in CI without editing this file.
DRIVE_ROOT = os.environ.get(
    "PRIMATE_DATA_ROOT",
    "/content/drive/MyDrive/primates-data",
)
# In the current Drive layout the species/ and background/ folders live
# directly under DRIVE_ROOT, so AUDIO_ROOT == DRIVE_ROOT by default.
AUDIO_ROOT = os.environ.get("PRIMATE_AUDIO_ROOT", DRIVE_ROOT)
LONG_AUDIO_ROOT = os.environ.get(
    "PRIMATE_LONG_AUDIO_ROOT", os.path.join(DRIVE_ROOT, "long_audio")
)

# SPECIES CONFIGURATION
# Cercopithecus nictitans call types are merged into a single "Cernic" class
# for presence detection. Each entry can be a single folder (str) or a list of
# folders whose clips are pooled under one label.
SPECIES_FOLDERS = {
    'Cernic': [
        'species/CERNIC putty-nose 2s',
        'species/CERNIC hacks',
        'species/CERNIC keks',
        'species/CERNIC pyows',
        # Real Cernic calls that were wrongly mined into Background by the
        # auto-cleanup loop (the model fired high-confidence Cernic on a
        # dev-station window, it was assumed a false positive, but human review
        # confirmed a genuine putty-nose call). Recovered here as positives.
        # Label safety: every clip is human-verified AND confirmed to originate
        # from a dev station (IPA1-18 / Makokou short-term), never the held-out
        # IPA19/20 -- so no test-station audio leaks into training. Safe to list
        # before it exists; scan_audio_files() warns and skips a missing folder.
        'species/CERNIC field_confirmed',
    ],
    'Colobus_guereza': 'species/Colobus guereza 2s windows',
    # Dedicated hard-negative class for the low-frequency forest sound that the
    # model repeatedly mis-fires as Colobus (pulsed, <1 kHz, morphologically
    # close to a guereza roar). These clips are ALL the Colobus false positives
    # mined by the auto-cleanup loop across dev stations -- human-reviewed so no
    # real Cernic leaks in (genuine Cernic found during review was deleted or
    # recovered into 'CERNIC field_confirmed'). Giving the confuser its own
    # softmax output forces the model to learn the Colobus-vs-confuser boundary
    # explicitly instead of drowning these few hundred hard negatives inside the
    # huge generic Background class (which V9 proved is not enough). At detection
    # time this class is folded into the Background group (see DETECTION_GROUPS)
    # so it never produces a detection.
    'Colobus_confuser': 'species/Colobus_confuser',
}

# Background noise folders (will be combined into single "Background" class)
# The last entry accumulates hard negatives from the auto-cleanup loop.
# To add impulsive-noise negatives (gunshots, branch-snaps) in a later round,
# create a 'background/impulsive_noise' folder and uncomment the line below.
BACKGROUND_FOLDERS = [
    'background/background noise Clips 5sec',
    'background/Cercocebus torquatus Clips 5s',
    'background/wrong classified',
    'background/Pan troglodytes Clips 5sec',
    # 'background/impulsive_noise',
    'outputs/auto_cleanup/auto_flagged_fp',
    # Confirmed field false positives mined from dev stations (IPA1-18) with
    # scripts/mine_field_negatives.py. Distribution-matched hard negatives (real
    # forest birds/insects/sawing/speech recorded by the same AudioMoth). Label
    # safety: all Colobus clips are taken (no real Colobus at any dev station),
    # but Cernic clips are YAMNet-gated -- only kept when an independent tagger
    # calls the window bird/insect/etc., so real putty-nose calls are never
    # pulled into Background. The held-out test stations IPA19/20 never feed in.
    # Safe to list before it exists -- scan_audio_files() warns and skips a
    # missing folder.
    'background/field_fp_negatives',
]

# AUDIO PARAMETERS
SAMPLE_RATE = 44100  # Hz
CLIP_DURATION = 2.0  # seconds — length of every TRAINING clip
# SLIDING-WINDOW DETECTION (preprocessing.extract_sliding_windows)
# A long field recording is sliced into fixed-length windows that are each
# classified independently, then high-confidence runs are merged into one
# detection (see detection.detect_in_long_audio).
#   WINDOW_SIZE   = length of each window, in seconds. Kept identical to
#                   CLIP_DURATION so the model sees the same 2 s input
#                   distribution it was trained on.
#   WINDOW_STRIDE = how far the window advances each step. 1.0 s on a 2.0 s
#                   window = 50% overlap, so a call that straddles a window
#                   boundary (e.g. sitting across [0-2s] and [2-4s]) is still
#                   fully captured by the overlapping [1-3s] window instead of
#                   being split in half and missed. Smaller stride = finer time
#                   resolution but more windows (slower); larger stride = faster
#                   but higher risk of clipping a call across the boundary.
WINDOW_SIZE = 2.0  # seconds (one detection window == one training clip length)
WINDOW_STRIDE = 1.0  # seconds (50% overlap so boundary-straddling calls aren't lost)

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

# COLOBUS HIGH-FREQUENCY NUISANCE AUGMENTATION (V12)
# The curated Colobus reference clips carry incidental high-frequency bird/insect
# energy. Because that high-freq content correlates with the Colobus label during
# training, the model learned to fire on high-freq TEXTURE and confused forest
# insects/birds (which sit at 2-5 kHz) with Colobus in the field, even after the
# V11 frequency-position head. To break that spurious correlation, every Colobus
# training clip also yields COLOBUS_HF_AUG_COUNT extra variants whose band ABOVE
# COLOBUS_HF_CUTOFF_HZ is replaced with high-freq content from random background
# clips, while the low-frequency roar (the true, invariant Colobus signature) is
# left untouched. With the high band decorrelated from the label, the model is
# forced to key on the low-frequency roar. Applied ONLY to the class named below
# (Cernic's discriminative energy IS high-freq and must be preserved).
COLOBUS_HF_AUG_CLASS = 'Colobus_guereza'
COLOBUS_HF_CUTOFF_HZ = 1500   # roar lives below this; randomize everything above
COLOBUS_HF_AUG_COUNT = 2      # extra high-freq-randomized variants per Colobus clip

# Geometric augmentation parameters
CHOP_RANGE = (0.1, 0.3)  # Crop 10-30% from edges
TRANSLATE_RANGE = (-20, 20)  # Frequency bins to shift

# TRAIN/VALIDATION SPLIT
VALIDATION_SPLIT = 0.2  # 20% for validation
RANDOM_SEED = 42

# MODEL PARAMETERS
MODEL_NAME = 'VGG19'
PRETRAINED_WEIGHTS = 'imagenet'
# Pooling head applied to the VGG19 feature map before the dense classifier:
#   'gap'        -> GlobalAveragePooling2D (averages away both frequency and
#                   time; the original head)
#   'freq_bands' -> low/mid/high frequency-band pooling (keeps frequency, V6)
#   'temporal'   -> frequency-pool + 1D-conv over time (keeps WHEN energy
#                   occurs; targets the Cernic-vs-insect/sawing confusion, V7)
#   'temporal_freq' -> per-band Conv1D + cross-band Conv1D + BiLSTM (keeps both
#                   WHEN and WHERE energy occurs; the PRODUCTION V10 head)
#   'temporal_freqpos' -> temporal_freq plus an explicit frequency-coordinate
#                   channel (CoordConv) fused into the feature map before the
#                   band split, so each call texture is tagged with its absolute
#                   frequency. Targets the Colobus(low)-vs-bird(high) confusion
#                   that the position-blind band split leaves unresolved (V11/V12)
# Overridable via the PRIMATE_MODEL_POOLING env var so the standard training
# pipeline can switch heads without editing code. The code default is 'gap';
# set PRIMATE_MODEL_POOLING=temporal_freqpos to reproduce the published V12
# model (use temporal_freq for the earlier V10 model).
MODEL_POOLING = os.environ.get('PRIMATE_MODEL_POOLING', 'gap')
FREEZE_BASE_LAYERS = True  # Freeze VGG19 base layers initially
UNFREEZE_LAST_N_BLOCKS = 2  # Stage-2 fine-tuning unfreezes the last 2 blocks
                            # of the block4_conv4-truncated base (block3, block4);
                            # this is the value used for the published V11/V12 model.

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001
DROPOUT_RATE = 0.5

# Early stopping
PATIENCE = 10  # Stop if validation loss doesn't improve for N epochs
MIN_DELTA = 0.001  # Minimum change to qualify as improvement

# DETECTION PARAMETERS
DETECTION_CONFIDENCE_THRESHOLD = 0.4  # Only keep detections above this
NMS_IOU_THRESHOLD = 0.5  # Non-maximum suppression overlap threshold

# LOW-FREQUENCY SPECTRAL-ENERGY GATE (post-processing for Colobus detections)
# A detected Colobus clip is kept only if the fraction of its spectral energy
# below LOWFREQ_GATE_CUTOFF (within the FMIN-FMAX band) is at least
# LOWFREQ_GATE_THRESHOLD. Real C. guereza roars are overwhelmingly low-frequency
# (p5 ~ 0.41 on the 617 reference clips), whereas the dominant out-of-distribution
# false positives (insects, cicadas at 2-5 kHz) are high-frequency (median ~0.01),
# so the gate removes the latter without touching real calls. It runs at
# detection time and adds a `low_freq_ratio` column to every Colobus detection,
# which doubles as a RANKING signal: sorting detections by this ratio surfaces the
# genuine low-frequency roar candidates (high ratio) for manual review and pushes
# the insect false positives (near-zero ratio) to the bottom. No retraining needed.
#
# The gate cannot create true positives -- it only removes false ones; the model
# must still fire on a real roar in the first place (that is what the V12
# high-frequency-nuisance augmentation above is for). Calibrate LOWFREQ_GATE_THRESHOLD
# with detection.lowfreq_energy_ratio (NOT the ad-hoc <1 kHz / full-spectrum metric)
# so the number matches the deployed gate; pick it below the reference-clip p5 and
# above the field false-positive p95 to keep real calls while cutting insects.
LOWFREQ_GATE_ENABLED = True    # apply the gate inside detect_in_long_audio
LOWFREQ_GATE_CUTOFF = 1500     # Hz
LOWFREQ_GATE_THRESHOLD = 0.20  # calibrated: FP max=0.092, Colobus p05=0.261;
                               # 0.20 cuts 100% FP, keeps 97.6% TP (gap 0.11 wide)

# TIME FILTER FOR FIELD RECORDINGS
# Coarse, FILE-LEVEL filter (it does NOT trim audio — it only decides which
# whole recordings to process). The recording's start time is parsed from its
# filename (e.g. "S20210225T065943" -> 06:59) and the file is kept only if that
# start time falls within [TIME_FILTER_START, TIME_FILTER_END] (inclusive).
#
# WHEN TO TURN IT ON (time_filter=True in get_ipa_station_files):
#   Production / survey runs. Putty-nose and Colobus call mainly in the early
#   morning, so restricting to the dawn window skips most of the day's audio —
#   far less compute and fewer false positives. Use the SAME window for every
#   station so per-station detection counts are comparable in the paper.
# WHEN TO TURN IT OFF (time_filter=False):
#   Debugging / recovering missed calls / auditing one station's full-day
#   behaviour (e.g. per-station spot-checks). Processes every recording.
#
# Set either bound to None to disable filtering entirely.
TIME_FILTER_START = "05:30"
TIME_FILTER_END = "10:30"

# IPA STATION CONFIGURATION
# Path to the root containing IPA station folders (IPA1ST, IPA2ST, ...)
IPA_ROOT = os.environ.get(
    "PRIMATE_IPA_ROOT",
    os.path.join(DRIVE_ROOT, "field_recordings"),
)

# OUTPUT PATHS
OUTPUT_ROOT = os.environ.get("PRIMATE_OUTPUT_ROOT", os.path.join(DRIVE_ROOT, "outputs"))
PROCESSED_DATA_DIR = os.path.join(OUTPUT_ROOT, "processed_data")
MODEL_SAVE_DIR = os.path.join(OUTPUT_ROOT, "models")
DETECTION_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "detections")
VISUALIZATION_DIR = os.path.join(OUTPUT_ROOT, "visualizations")

# Create output directories if they don't exist. Wrapped in try/except so that
# importing this module never crashes on read-only filesystems (e.g. CI runners
# inspecting the package without access to the data drive).
for directory in [OUTPUT_ROOT, PROCESSED_DATA_DIR, MODEL_SAVE_DIR,
                  DETECTION_OUTPUT_DIR, VISUALIZATION_DIR]:
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as exc:
        print(f" Warning: could not create output directory {directory}: {exc}")


# DERIVED PARAMETERS
N_CLASSES = len(SPECIES_FOLDERS) + 1  # +1 for Background class
CLASS_NAMES = list(SPECIES_FOLDERS.keys()) + ['Background']

# DETECTION GROUPING
# With the merged Cernic class each model class maps directly to its own
# detection group — no probability aggregation needed at detection time.
DETECTION_GROUPS = {
    'Cernic': 'Cernic',
    'Colobus_guereza': 'Colobus_guereza',
    # The confuser is a trained class but NOT a detection target: route its
    # softmax mass into the Background group so a window the model calls
    # "confuser" is excluded from detections exactly like Background. Crucially
    # this keeps the confuser probability OUT of the Colobus_guereza group, so a
    # real guereza window is no longer inflated by confuser energy.
    'Colobus_confuser': 'Background',
    'Background': 'Background',
}

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
