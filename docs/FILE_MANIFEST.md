# 📦 Primate Vocalization Detection Pipeline - File Manifest

## 📄 Files Overview

This package contains a complete, reproducible pipeline for detecting primate vocalizations in long audio recordings.

---

## 🗂️ Core Modules (8 Python files)

### 1. `config.py` (7.2 KB)
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

### 2. `data_loader.py` (7.4 KB)
**Purpose**: Load and scan audio files  
**Key functions**:
- `load_species_data()` - Load all primate clips
- `load_background_data()` - Load background noise
- `load_long_audio()` - Load files for detection
- `get_long_audio_files()` - List all long audio files

**Automatic**: No editing needed - reads from config

---

### 3. `preprocessing.py` (6.8 KB)
**Purpose**: Convert audio to mel-spectrograms  
**Key functions**:
- `audio_to_melspectrogram()` - Core conversion
- `preprocess_audio()` - Complete pipeline
- `extract_sliding_windows()` - For detection
- `visualize_spectrogram()` - View spectrograms

**Automatic**: Uses parameters from config

---

### 4. `augmentation.py` (11 KB)
**Purpose**: Data augmentation strategies  
**Key functions**:
- `add_background_noise()` - Mix with background
- `time_chop()` - Crop time axis
- `freq_chop()` - Crop frequency axis
- `translate()` - Frequency shift
- `augment_dataset()` - Full augmentation pipeline

**Configuration**: Edit augmentation multipliers in config

---

### 5. `model.py` (7.7 KB)
**Purpose**: VGG19-based model definition  
**Key functions**:
- `build_model()` - Create model architecture
- `compile_model()` - Set optimizer and loss
- `get_callbacks()` - Training callbacks
- `load_trained_model()` - Load saved model

**Automatic**: Adapts to number of classes from config

---

### 6. `train.py` (11 KB)
**Purpose**: Complete training pipeline  
**Key functions**:
- `prepare_dataset()` - Load and augment data
- `train_model()` - Train the model
- `evaluate_model()` - Evaluate performance
- `run_complete_training_pipeline()` - One-command training

**Usage**: Called from notebook

---

### 7. `detection.py` (8.6 KB)
**Purpose**: Detect in long audio files  
**Key functions**:
- `detect_in_long_audio()` - Detect in one file
- `process_all_long_audio_files()` - Batch processing
- `apply_nms()` - Non-Maximum Suppression
- `save_detections()` - Save to CSV

**Usage**: Called from notebook after training

---

### 8. `utils.py` (12 KB)
**Purpose**: Visualization and analysis  
**Key functions**:
- `visualize_detection_results()` - Plot waveform + detections
- `create_detection_summary_report()` - Summary CSV
- `extract_detected_audio_clips()` - Extract clips
- `print_detection_statistics()` - Stats report

**Usage**: Called from notebook for analysis

---

## 📓 Main Notebook

### `main_pipeline.ipynb` (13 KB)
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

## 📚 Documentation (3 files)

### `README.md` (8.4 KB)
Complete documentation covering:
- Project overview
- Setup instructions
- Configuration details
- Pipeline workflow
- Advanced usage
- Troubleshooting

### `QUICK_START.md` (3.5 KB)
Fast-track guide with:
- 3-step setup
- Common issues
- Quick tips

### `FILE_MANIFEST.md` (This file)
Description of all files

---

## 🔄 Workflow Summary

```
1. Edit config.py → Set paths and parameters
2. Upload all .py files to Colab
3. Open main_pipeline.ipynb
4. Run cells sequentially
5. View results in outputs/ folder
```

---

## 📊 Expected File Sizes

```
config.py          7.2 KB
data_loader.py     7.4 KB
preprocessing.py   6.8 KB
augmentation.py   11.0 KB
model.py           7.7 KB
train.py          11.0 KB
detection.py       8.6 KB
utils.py          12.0 KB
main_pipeline.ipynb 13 KB
README.md          8.4 KB
QUICK_START.md     3.5 KB
─────────────────────────
Total:           ~96 KB
```

---

## 🎯 Which Files to Edit?

### Always Edit:
- ✏️ `config.py` - Update paths and add new species

### Usually Don't Edit:
- 🔒 All other `.py` files (unless customizing algorithms)
- 🔒 `main_pipeline.ipynb` (unless adding custom analysis)

### Documentation:
- 📖 README files are for reference only

---

## 🚀 Minimal Setup

To get started, you only need to:
1. Edit `config.py` (lines 12-26)
2. Upload all files to Colab
3. Run `main_pipeline.ipynb`

Everything else is automatic!

---

## 💾 After Training

Your model and results will be saved to:
```
drive/MyDrive/chimp-audio/outputs/
├── models/           # Trained models
├── detections/       # Detection results (CSV)
└── visualizations/   # Plots
```

---

## 🔄 Version Control

To save different experimental configurations:
1. Copy `config.py` → `config_experiment1.py`
2. Modify parameters
3. In notebook, import different config:
   ```python
   import config_experiment1 as config
   ```

---

## ✅ Checklist Before Running

- [ ] All `.py` files uploaded to Colab
- [ ] `main_pipeline.ipynb` uploaded
- [ ] Google Drive mounted in Colab
- [ ] Paths in `config.py` match your Drive structure
- [ ] Species folders contain `.wav` files
- [ ] GPU enabled in Colab (Runtime → Change runtime type → GPU)

---

**You're all set! 🎉 Open `QUICK_START.md` to begin!**
