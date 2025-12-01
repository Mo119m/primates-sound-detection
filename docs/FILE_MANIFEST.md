# Primate Vocalization Detection Pipeline - File Manifest

## Files Overview

This package contains a complete, reproducible pipeline for detecting primate vocalizations in long audio recordings.

---

## Core Modules (8 Python files)

### 1. `config.py` 
**Purpose**: Central configuration file  
**Contains**:
- All file paths (Google Drive structure)
- Species and background folder names
- Audio parameters (sample rate, clip duration)
- Mel-spectrogram settings
- Data augmentation configuration
- Model hyperparameters
- Detection thresholds

**When to edit**: 
- Adding new species
- Changing folder locations
- Adjusting parameters

---

### 2. `data_loader.py` 
**Purpose**: Load and scan audio files  
**Key functions**:
- `load_species_data()` - Load all primate clips
- `load_background_data()` - Load background noise
- `load_long_audio()` - Load files for detection
- `get_long_audio_files()` - List all long audio files

**Automatic**: No editing needed - reads from config

---

### 3. `preprocessing.py` 
**Purpose**: Convert audio to mel-spectrograms  
**Key functions**:
- `audio_to_melspectrogram()` - Core conversion
- `preprocess_audio()` - Complete pipeline
- `extract_sliding_windows()` - For detection
- `visualize_spectrogram()` - View spectrograms

**Automatic**: Uses parameters from config

---

### 4. `augmentation.py` 
**Purpose**: Data augmentation strategies  
**Key functions**:
- `add_background_noise()` - Mix with background
- `time_chop()` - Crop time axis
- `freq_chop()` - Crop frequency axis
- `translate()` - Frequency shift
- `augment_dataset()` - Full augmentation pipeline

**Configuration**: Edit augmentation multipliers in config

---

### 5. `model.py` 
**Purpose**: VGG19-based model definition  
**Key functions**:
- `build_model()` - Create model architecture
- `compile_model()` - Set optimizer and loss
- `get_callbacks()` - Training callbacks
- `load_trained_model()` - Load saved model

**Automatic**: Adapts to number of classes from config

---

### 6. `train.py` 
**Purpose**: Complete training pipeline  
**Key functions**:
- `prepare_dataset()` - Load and augment data
- `train_model()` - Train the model
- `evaluate_model()` - Evaluate performance
- `run_complete_training_pipeline()` - One-command training

**Usage**: Called from notebook

---

### 7. `detection.py` 
**Purpose**: Detect in long audio files  
**Key functions**:
- `detect_in_long_audio()` - Detect in one file
- `process_all_long_audio_files()` - Batch processing
- `apply_nms()` - Non-Maximum Suppression
- `save_detections()` - Save to CSV

**Usage**: Called from notebook after training

---

### 8. `utils.py`
**Purpose**: Visualization and analysis  
**Key functions**:
- `visualize_detection_results()` - Plot waveform + detections
- `create_detection_summary_report()` - Summary CSV
- `extract_detected_audio_clips()` - Extract clips
- `print_detection_statistics()` - Stats report

**Usage**: Called from notebook for analysis

---

## Main Notebook

### `main_pipeline.ipynb` 
**Purpose**: Interactive pipeline execution  
**Sections**:
1. Setup & Installation
2. Configuration
3. Data Loading
4. Training Pipeline
5. Detection on Long Audio
6. Analysis & Reporting
7. Optional Clip Extraction
8. Threshold Adjustment

**Usage**: Run sequentially in Google Colab

---

## Documentation 

### `README.md` 
Complete documentation covering:
- Project overview
- Setup instructions
- Configuration details
- Pipeline workflow
- Advanced usage
- Troubleshooting

### `QUICK_START.md` 
Fast-track guide with:
- 3-step setup
- Common issues
- Quick tips

### `FILE_MANIFEST.md` (This file)
Description of all files

---

## Workflow Summary

```
1. Edit config.py → Set paths and parameters
2. Upload all .py files to Colab
3. Open main_pipeline.ipynb
4. Run cells sequentially
5. View results in outputs/ folder
```

---

## Which Files to Edit?

### Always Edit:
-  `config.py` - Update paths and add new species

### Usually Don't Edit:
- All other `.py` files (unless customizing algorithms)
- `main_pipeline.ipynb` (unless adding custom analysis)

---

##  Minimal Setup

To get started, only need to:
1. Edit `config.py` (lines 12-26)
2. Upload all files to Colab
3. Run `main_pipeline.ipynb`

Everything else is automatic

---

## After Training

Model and results will be saved to:
```
drive/MyDrive/chimp-audio/outputs/
├── models/           # Trained models
├── detections/       # Detection results (CSV)
└── visualizations/   # Plots
```

---


**Open `QUICK_START.md` to begin!**
