# Quick Start Guide

This guide provides rapid setup instructions for the Primate Vocalization Detection Pipeline. 

## Prerequisites
- Google Drive with organized audio data


## Installation

### Option 1: Google Colab (Recommended)

- As shown in readme

### Option 2: Local Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/primate-vocalization-detection.git
cd primate-vocalization-detection
```

2. Create virtual environment:

```bash
python -m venv venv
source venv/bin/activate  
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Organization

Ensure Google Drive follows this structure:

```
drive/MyDrive/chimp-audio/
├── audio/
│   ├── Cercocebus torquatus hack 5s/    # Training clips for species 1
│   ├── Colobus guereza Clips 5s/        # Training clips for species 2
│   ├── background noise Clips 5sec/     # Environmental sounds
│   └── wrong classified/                # Additional negative examples
└── long_audio/
    └── *.wav                            # Long recordings for detection
```

All training clips should be approximately 5 seconds in duration. Long audio files can be any length but typically 10-30 minutes.

## Configuration

Edit `src/config.py` to match your Google Drive structure:

```python
# Lines 12-14: Update these paths
DRIVE_ROOT = "/content/drive/MyDrive/chimp-audio"
AUDIO_ROOT = os.path.join(DRIVE_ROOT, "audio")
LONG_AUDIO_ROOT = os.path.join(DRIVE_ROOT, "long_audio")
```

Verify species folder names match your data:

```python
# Lines 17-21
SPECIES_FOLDERS = {
    'Cercocebus_torquatus': 'Cercocebus torquatus hack 5s',
    'Colobus_guereza': 'Colobus guereza Clips 5s',
}
```

## Basic Workflow

### Step 1: Import modules

```python
import sys
sys.path.append('src')

from src import config
from src import data_loader
from src import train
from src import detection
from src import utils

config.print_config_summary()
```

### Step 2: Train the model

```python
trained_model = train.run_complete_training_pipeline()
```

Expected time: 30-60 minutes on GPU

### Step 3: Run detection

```python
long_audio_files = data_loader.get_long_audio_files()
first_audio = long_audio_files[0]

detections_df = detection.detect_in_long_audio(
    trained_model,
    first_audio,
    confidence_threshold=0.7
)

print(f"Found {len(detections_df)} detections")
```

### Step 4: Visualize results

```python
utils.visualize_detection_results(
    first_audio,
    detections_df,
    show_spectrogram=True
)
```

### Step 5: Save results

```python
detection.save_detections(detections_df, os.path.basename(first_audio))
```

Results are saved to `outputs/detections/` in CSV format.

## Adding New Species

To add a new primate species:

1. Add training clips to Google Drive: `audio/new_species_folder/`

2. Edit `src/config.py` (lines 17-21):

```python
SPECIES_FOLDERS = {
    'Cercocebus_torquatus': 'Cercocebus torquatus hack 5s',
    'Colobus_guereza': 'Colobus guereza Clips 5s',
    'New_Species': 'new_species_folder',  # Add this line
}
```

3. Retrain the model:

```python
import importlib
importlib.reload(config)
new_model = train.run_complete_training_pipeline()
```

The pipeline automatically adjusts the model architecture and data processing.

## Common Adjustments

### Reduce memory usage

Edit `src/config.py` line 89:

```python
BATCH_SIZE = 16  # Reduce from 32
```

### Adjust detection sensitivity

Edit `src/config.py` line 101:

```python
DETECTION_CONFIDENCE_THRESHOLD = 0.8  # Increase for fewer detections
DETECTION_CONFIDENCE_THRESHOLD = 0.6  # Decrease for more detections
```

### Modify augmentation

Edit `src/config.py` lines 62-68:

```python
AUGMENTATION_CONFIG = {
    'original': 1,
    'background_noise_mix': 5,  # Increase for more augmentation
    'time_chop': 2,
    'freq_chop': 2,
    'translate': 2,
}
```

## Troubleshooting (also working now. looking for more ways to solve)

### Poor detection results

1. Check training accuracy: Should be above 80%
2. Reduce confidence threshold
3. Implement hard negative mining workflow (see scripts/run_hard_negative_mining.py)

## Output Files

After running the pipeline, outputs are organized in `drive/MyDrive/chimp-audio/outputs/`:

Training outputs:
- `models/best_model.h5` - Trained model
- `models/training_history.json` - Training metrics
- `models/training_history.png` - Training curves

Detection outputs:
- `detections/*_detections.csv` - Detection results per file
- `detections/detection_summary.csv` - Summary across files
- `visualizations/*_visualization.png` - Plots

## Next Steps

After successful basic workflow:

1. Process all long audio files:

```python
all_detections = detection.process_all_long_audio_files(trained_model)
```

2. Generate comprehensive reports:

```python
utils.print_detection_statistics(all_detections)
summary_df = utils.create_detection_summary_report(all_detections)
```

3. Extract detected audio clips for verification:

```python
clips_dir = 'outputs/detected_clips'
utils.extract_detected_audio_clips(first_audio, detections_df, clips_dir)
```

4. Implement hard negative mining to improve model (see SETUP_TUTORIAL.md)



