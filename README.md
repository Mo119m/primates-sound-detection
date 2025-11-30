# Primate Vocalization Detection Pipeline

A reproducible machine learning system for automated detection of primate vocalizations in long-term rainforest audio recordings. This pipeline addresses the challenge of efficiently identifying species-specific calls in hours of field recordings through transfer learning and targeted data augmentation.

## Overview

This project implements an end-to-end deep learning pipeline for detecting vocalizations of two primate species (Cercocebus torquatus and Colobus guereza) in 30-minute rainforest audio recordings from Makokou, Gabon. The system uses VGG19 transfer learning with mel-spectrogram preprocessing and addresses limited training data through strategic data augmentation and hard negative mining.

### Key Features

- Modular design enabling easy addition of new species
- Reproducible workflow with fixed random seeds
- Centralized configuration system
- GPU-optimized for Google Colab environment
- Comprehensive visualization and analysis tools
- Hard negative mining to address domain gap issues

### Research Context

Field recordings from tropical rainforests contain complex acoustic environments with overlapping vocalizations from multiple species, environmental sounds, and varying signal-to-noise ratios. Manual review of such recordings is time-intensive, creating a bottleneck for ecological research. This pipeline was developed to assist researchers in efficiently identifying primate calls, enabling faster analysis of long-term acoustic monitoring data.

## Installation

### Clone the repository

```bash
git clone https://github.com/yourusername/primate-vocalization-detection.git
cd primate-vocalization-detection
```

### Install dependencies

```bash
pip install -r requirements.txt
```

For Google Colab environment:

```python
!pip install -q librosa soundfile tensorflow scikit-learn pandas matplotlib
```

### Data structure

The pipeline expects audio data organized in Google Drive with the following structure:

```
chimp-audio/
├── audio/
│   ├── Cercocebus torquatus hack 5s/    # Species 1 training clips (5 seconds)
│   ├── Colobus guereza Clips 5s/        # Species 2 training clips (5 seconds)
│   ├── background noise Clips 5sec/     # Background/environmental sounds
│   └── wrong classified/                # Additional negative examples
└── long_audio/
    └── *.wav                            # Long recordings for detection (typically 30 minutes)
```

## Project Structure

```
primate-vocalization-detection/
├── src/                          # Source code modules
│   ├── config.py                # Configuration parameters
│   ├── data_loader.py           # Audio file loading
│   ├── preprocessing.py         # Mel-spectrogram conversion
│   ├── augmentation.py          # Data augmentation strategies
│   ├── model.py                 # VGG19-based model definition
│   ├── train.py                 # Training pipeline
│   ├── detection.py             # Detection in long audio files
│   └── utils.py                 # Visualization and analysis
├── scripts/                     # Utility scripts
│   └── run_hard_negative_mining.py
├── notebooks/                   # Jupyter notebooks
│   └── main_pipeline.ipynb     # Main workflow
├── docs/                        # Documentation
└── requirements.txt            # Python dependencies
```

## Quick Start

### Basic usage in Jupyter notebook

```python
import sys
sys.path.append('src')

from src import config
from src import train
from src import detection

# Configure paths in config.py first
config.print_config_summary()

# Train model
trained_model = train.run_complete_training_pipeline()

# Detect in long audio
from src import data_loader
long_audio_files = data_loader.get_long_audio_files()
detections = detection.detect_in_long_audio(trained_model, long_audio_files[0])
```

### Google Colab workflow

1. Mount Google Drive and navigate to project directory
2. Open `notebooks/main_pipeline.ipynb`
3. Run cells sequentially
4. Results will be saved to `outputs/` directory in Google Drive

For detailed instructions, see `docs/SETUP_TUTORIAL.md`.

## Configuration

All parameters are centralized in `src/config.py`. Key configuration sections include:

### Data paths

```python
DRIVE_ROOT = "/content/drive/MyDrive/chimp-audio"
AUDIO_ROOT = os.path.join(DRIVE_ROOT, "audio")
LONG_AUDIO_ROOT = os.path.join(DRIVE_ROOT, "long_audio")
```

### Species configuration

```python
SPECIES_FOLDERS = {
    'Cercocebus_torquatus': 'Cercocebus torquatus hack 5s',
    'Colobus_guereza': 'Colobus guereza Clips 5s',
}
```

### Audio processing parameters

```python
SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
FMIN = 20
FMAX = 8000
```

### Data augmentation scheme

```python
AUGMENTATION_CONFIG = {
    'original': 1,
    'background_noise_mix': 3,
    'time_chop': 1,
    'freq_chop': 1,
    'translate': 1,
}
```

This configuration produces a 7x augmentation multiplier for each training sample.

### Model hyperparameters

```python
MODEL_NAME = 'VGG19'
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001
DROPOUT_RATE = 0.5
```

### Detection parameters

```python
DETECTION_CONFIDENCE_THRESHOLD = 0.7
NMS_IOU_THRESHOLD = 0.5
WINDOW_SIZE = 5.0
WINDOW_STRIDE = 2.5
```

## Methodology

### Data augmentation

The pipeline implements conservative data augmentation adapted from tropical-stethoscope methods:

1. Background noise mixing with random SNR
2. Time-domain cropping (10-30% from edges)
3. Frequency-domain cropping (10-30% from edges)
4. Frequency translation (±20 mel bins)

Each primate vocalization sample generates 7 augmented versions, while background samples remain unaugmented to preserve class balance.

### Model architecture

The model uses transfer learning from VGG19 pre-trained on ImageNet:

```
Input (224×224×3 mel-spectrogram)
    ↓
VGG19 base (frozen, ImageNet weights)
    ↓
GlobalAveragePooling2D
    ↓
Dense(512, relu) + Dropout(0.5)
    ↓
Dense(256, relu) + Dropout(0.5)
    ↓
Dense(N_CLASSES, softmax)
```

### Detection process

Long audio files are processed using a sliding window approach:

1. Extract 5-second windows with 2.5-second stride (50% overlap)
2. Convert each window to mel-spectrogram
3. Apply model prediction
4. Filter detections by confidence threshold
5. Apply Non-Maximum Suppression (NMS) to remove overlapping detections
6. Export results to CSV with timestamps

### Hard negative mining

To address the domain gap between clean training data and complex field recordings, the pipeline includes a hard negative mining workflow:

1. Extract samples where the model predicts primate calls with medium confidence (0.5-0.85)
2. Manual verification to identify false positives (typically bird calls)
3. Incorporation of verified false positives as hard negatives in retraining
4. Iterative improvement of model discrimination

## Adding New Species

The modular design allows straightforward addition of new species:

1. Add training clips to Google Drive: `audio/new_species_clips/`
2. Edit `src/config.py`:

```python
SPECIES_FOLDERS = {
    'Cercocebus_torquatus': 'Cercocebus torquatus hack 5s',
    'Colobus_guereza': 'Colobus guereza Clips 5s',
    'New_Species': 'new_species_clips',
}
```

3. Re-run training pipeline

The system automatically adjusts the model architecture, data loading, and visualization components.

## Output Files

After running the pipeline, results are organized in the `outputs/` directory:

```
outputs/
├── models/
│   ├── best_model.h5              # Trained model weights
│   ├── training_history.json      # Training metrics
│   └── training_history.png       # Training curves
├── detections/
│   ├── [filename]_detections.csv  # Detection results per file
│   └── detection_summary.csv      # Summary across all files
├── visualizations/
│   └── [filename]_visualization.png
└── detected_clips/
    ├── Cercocebus_torquatus/
    └── Colobus_guereza/
```

Detection CSV files contain columns: start_time, end_time, species, confidence.

## Expected Performance

### Training data distribution (after augmentation)

- Cercocebus torquatus: 182 clips → 1,274 samples
- Colobus guereza: 172 clips → 1,204 samples
- Background: 1,035 samples (no augmentation)
- Total: approximately 3,500 samples

### Model performance

- Training time: 30-60 minutes (GPU-dependent)
- Expected validation accuracy: 85-95% on clean data
- Detection speed: 1-2 minutes per 30-minute audio file

Note that validation accuracy on augmented clean data does not directly predict real-world detection performance due to domain gap. Hard negative mining addresses this limitation.

## Known Limitations

1. **Domain gap**: Model trained on clean clips may overestimate primate calls in complex soundscapes containing bird vocalizations
2. **Class imbalance**: Detection results may show bias toward more frequently augmented classes
3. **Temporal context**: 5-second windows may truncate longer vocalizations or miss relevant context
4. **Generalization**: Model performance may degrade on recordings from different geographic locations or recording equipment

Hard negative mining and confidence threshold tuning help mitigate these issues.

## Troubleshooting

### GPU out of memory

Reduce batch size in `src/config.py`:

```python
BATCH_SIZE = 16  # or 8
```

### Excessive false positives

Increase confidence threshold:

```python
DETECTION_CONFIDENCE_THRESHOLD = 0.8
```

Or implement hard negative mining workflow (see `scripts/run_hard_negative_mining.py`).

### Poor model convergence

Increase data augmentation multiplier or add more training data. Consider unfreezing VGG19 layers for fine-tuning:

```python
FREEZE_BASE_LAYERS = False
UNFREEZE_LAST_N_BLOCKS = 1
LEARNING_RATE = 0.00001  # Lower learning rate for fine-tuning
```

## References

### Methods

This pipeline builds on established methods in bioacoustics and deep learning:

- Transfer learning approach based on VGG19 (Simonyan & Zisserman, 2014)
- Data augmentation strategies adapted from tropical-stethoscope project
- Mel-spectrogram preprocessing using librosa (McFee et al., 2015)

### Dependencies

- TensorFlow 2.x: Deep learning framework
- librosa 0.10+: Audio processing
- scikit-learn: Machine learning utilities
- pandas: Data manipulation
- matplotlib: Visualization

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{primate_vocalization_detection_2024,
  author = {Mo},
  title = {Primate Vocalization Detection Pipeline: Automated Detection of Primate Calls in Rainforest Audio Recordings},
  year = {2024},
  url = {https://github.com/yourusername/primate-vocalization-detection},
  note = {Collaboration with Santiago (data provider) and Professor Claudia (supervisor)}
}
```

## Acknowledgments

This project was developed in collaboration with Santiago (data provider) and under the supervision of Professor Claudia. Audio recordings were collected from Makokou, Gabon as part of long-term rainforest monitoring research.

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Contact

For questions, issues, or collaboration opportunities, please open an issue on GitHub or contact the author directly.

## Version History

- v1.0.0 (2024-11): Initial release
  - Three-class detection (Cercocebus torquatus, Colobus guereza, Background)
  - Conservative augmentation scheme (7x multiplier)
  - VGG19 transfer learning
  - Hard negative mining workflow
  - Complete visualization and analysis pipeline
