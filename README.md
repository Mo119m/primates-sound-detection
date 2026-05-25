# Primate Vocalization Detection

Automated detection of primate calls in long rainforest field recordings from Makokou, Gabon. Uses VGG19 transfer learning on mel-spectrograms with a three-filter false-positive cleanup pipeline. Currently targets **Cercopithecus nictitans** (putty-nosed monkey) and **Colobus guereza**.

## Main Workflow

The pipeline has 5 stages. Each one below shows **exactly which function to call**.
Easiest path: open `main_pipeline_notebooks/run_in_colab.ipynb` and run the cells,
which call these same functions for you.

```
Step 1  Configure   ->  Step 2  Train  ->  Step 3  Detect  ->  Step 4  Clean up  ->  Step 5  Retrain
edit config.py          train.run_       detection.        auto_cleanup.          fold FPs into
                        complete_        process_all_      run_auto_cleanup()     Background, go to
                        training_        long_audio_                              Step 2 (iterate)
                        pipeline()       files()
```

```python
import sys; sys.path.insert(0, 'src')
import config, train, detection, auto_cleanup, model, data_loader
```

### Step 1 — Configure

Edit `src/config.py`: set `SPECIES_FOLDERS`, `BACKGROUND_FOLDERS`, and the data
paths (or set the `PRIMATE_*` environment variables). Check it loaded correctly:

```python
config.print_config_summary()
```

### Step 2 — Train the model

One call runs the whole training pipeline (load audio -> spectrograms ->
augmentation -> train -> evaluate). It saves `best_model.h5` to `outputs/models/`.

```python
trained_model = train.run_complete_training_pipeline()
```

> Want the individual steps instead? `train.prepare_dataset()` ->
> `train.train_model(...)` -> `train.evaluate_model(...)`.

### Step 3 — Detect in field recordings

Load the trained model, then run detection. Two options:

```python
model_obj = model.load_trained_model('outputs/models/best_model.h5')

# (a) one file -> DataFrame of detections
detections = detection.detect_in_long_audio(model_obj, '/path/to/recording.wav')

# (b) every file under LONG_AUDIO_ROOT -> {filename: DataFrame}, writes CSVs
all_detections = detection.process_all_long_audio_files(model_obj)
```

Or detect a whole IPA station from the command line (applies the time-of-day filter):

```bash
python scripts/run_detection_ipa.py --station IPA1ST
```

### Step 4 — Auto-cleanup false positives

Run the three filters (Mahalanobis OOD + YAMNet + temporal isolation) over the
detection CSVs. Returns clean vs. suspicious detections and saves hard-negative
clips to `outputs/auto_cleanup/auto_flagged_fp/`.

```python
result = auto_cleanup.run_auto_cleanup(detection_dir='outputs/detections/IPA1ST')
result['clean_df']       # detections that passed all filters
result['suspicious_df']  # flagged, with a flag_reason column
```

Or from the command line:

```bash
python scripts/run_auto_cleanup.py --detection-dir outputs/detections/IPA1ST
```

### Step 5 — Retrain with hard negatives (iterate)

Move the flagged clips from `auto_cleanup/auto_flagged_fp/` into a background
folder, add it to `BACKGROUND_FOLDERS` in `config.py`, then go back to **Step 2**.
Repeat 3-5 times until false positives drop off.

## Repository Structure

```
src/                           Core library modules
scripts/                       Command-line entry points
main_pipeline_notebooks/       Colab notebooks for training, detection, and cleanup
presentation_notebooks/        Figures and slides generation
```

## Source Modules (`src/`)

### config.py
All paths, parameters, and species definitions in one place. Edit this file to change species, audio settings, model hyperparameters, or output directories.

### data_loader.py
Load and manage audio files.

| Function | Description |
|---|---|
| `load_species_data()` | Load all species audio clips into a dictionary |
| `load_background_data()` | Load background noise clips from multiple folders |
| `load_audio_file()` | Load a single WAV file with padding/cropping to fixed length |
| `get_ipa_station_files()` | Get WAV files for an IPA station with optional time-of-day filtering |
| `get_long_audio_files()` | List all long audio files recursively |
| `filter_files_by_time()` | Keep only recordings within a time window |

### preprocessing.py
Convert audio waveforms to mel-spectrogram images for model input.

| Function | Description |
|---|---|
| `preprocess_audio()` | Full pipeline: audio waveform -> mel-spectrogram -> 224x224 RGB image |
| `extract_sliding_windows()` | Extract overlapping windows from long audio with timestamps |
| `batch_preprocess_audio()` | Preprocess multiple audio samples |
| `audio_to_melspectrogram()` | Convert waveform to mel-spectrogram in dB scale |

### augmentation.py
Data augmentation on mel-spectrograms to expand training data (7x multiplier).

| Function | Description |
|---|---|
| `augment_dataset()` | Augment entire dataset, producing X, y arrays and metadata |
| `add_background_noise()` | Mix spectrogram with background noise at random SNR |
| `time_chop()` | Randomly crop along time axis |
| `freq_chop()` | Randomly crop along frequency axis |
| `translate()` | Shift spectrogram in frequency |

### model.py
VGG19-based transfer learning model.

| Function | Description |
|---|---|
| `create_and_compile_model()` | Build VGG19 + custom head and compile |
| `load_trained_model()` | Load a saved `.h5` model |
| `unfreeze_base_model()` | Unfreeze last N VGG19 blocks for fine-tuning |
| `get_callbacks()` | EarlyStopping, ModelCheckpoint, ReduceLROnPlateau |

### train.py
End-to-end training pipeline.

| Function | Description |
|---|---|
| `run_complete_training_pipeline()` | Orchestrate the full workflow: load data -> augment -> train -> evaluate |
| `prepare_dataset()` | Load audio, convert to spectrograms, augment, split train/val |
| `calculate_class_weights()` | Compute balanced class weights for imbalanced data |
| `train_model()` | Train with class weights and callbacks |
| `evaluate_model()` | Per-class metrics and confusion matrix |

### detection.py
Sliding-window detection in long field recordings with probability grouping.

| Function | Description |
|---|---|
| `detect_in_long_audio()` | Run full detection on a single file (windows -> predict -> NMS -> CSV) |
| `process_all_long_audio_files()` | Detect across all files and aggregate results |
| `group_probabilities()` | Sum softmax scores across Cernic subtypes before thresholding |
| `sweep_thresholds()` | Apply multiple confidence thresholds to pre-computed predictions |
| `apply_nms()` | Non-Maximum Suppression to remove overlapping detections |
| `save_detections()` | Save detection results to CSV |

### auto_cleanup.py
Three-filter automatic false-positive cleanup.

| Function | Description |
|---|---|
| `run_auto_cleanup()` | Orchestrate all three filters and save results |
| `filter_mahalanobis()` | Flag detections whose features are out-of-distribution |
| `filter_yamnet()` | Flag detections whose YAMNet top class is non-primate |
| `filter_temporal_isolation()` | Flag isolated detections with no same-species neighbor nearby |
| `save_hard_negatives()` | Export strong false positives as WAV clips for retraining |

### utils.py
Visualization and analysis utilities.

| Function | Description |
|---|---|
| `visualize_detection_results()` | Plot waveform and spectrogram with detection overlays |
| `plot_confusion_matrix()` | Display confusion matrix |
| `extract_detected_audio_clips()` | Extract WAV clips for detected events |
| `create_detection_summary_report()` | Per-file per-species detection statistics |

## Scripts (`scripts/`)

| Script | Description |
|---|---|
| `run_detection_ipa.py` | Run detection on an IPA field recording station. Args: `--station`, `--model`, `--threshold`, `--no-time-filter` |
| `run_auto_cleanup.py` | Run the three-filter false-positive cleanup. Args: `--detection-dir`, `--model`, `--percentile`, `--isolation-window` |
| `run_hard_negative_mining.py` | Extract medium-confidence predictions as candidate false positives for retraining |
| `filter_recordings_by_time.py` | Copy only recordings within a time-of-day window (pre-upload filter) |
| `analyze_detections.py` | Per-species detection analysis and threshold suggestion helpers |

## Notebooks (`main_pipeline_notebooks/`)

| Notebook | Description |
|---|---|
| `run_in_colab.ipynb` | Full pipeline: setup, train, detect (start here) |
| `main_pipeline_updated.ipynb` | Detailed step-by-step training and evaluation |
| `auto_cleanup_false_positives.ipynb` | Run auto-cleanup interactively with visualization |

## Configuration

All parameters live in `src/config.py`. Key settings:

| Parameter | Default | Description |
|---|---|---|
| `SAMPLE_RATE` | 44100 Hz | Audio sample rate |
| `CLIP_DURATION` | 2.0 s | Training clip length |
| `WINDOW_SIZE` / `WINDOW_STRIDE` | 2.0 / 1.0 s | Detection sliding window |
| `N_MELS` | 128 | Mel-spectrogram frequency bins |
| `FMIN` / `FMAX` | 20 / 8000 Hz | Frequency range |
| `BATCH_SIZE` | 32 | Training batch size |
| `EPOCHS` | 50 | Max training epochs |
| `DETECTION_CONFIDENCE_THRESHOLD` | 0.4 | Minimum confidence for detections |
| `TIME_FILTER_START` / `END` | 05:30 / 10:30 | Field recording time window |

Override data paths via environment variables: `PRIMATE_DATA_ROOT`, `PRIMATE_AUDIO_ROOT`, `PRIMATE_LONG_AUDIO_ROOT`, `PRIMATE_IPA_ROOT`, `PRIMATE_OUTPUT_ROOT`.

## Data Layout

```
primates-data/                        (PRIMATE_DATA_ROOT)
  species/
    CERNIC hacks/                     Putty-nosed monkey hack calls
    CERNIC keks/                      Putty-nosed monkey kek calls
    CERNIC pyows/                     Putty-nosed monkey pyow calls
    Colobus guereza Clips 5s/         Colobus guereza calls
  background/
    background noise Clips 5sec/      Environmental noise
    Cercocebus torquatus Clips 5s/    Non-target species
    wrong classified/                 Misclassified examples
    Pan troglodytes Clips 5sec/       Non-target species
  field_recordings/
    IPA1ST/YYYYMMDD/*.wav             IPA station field recordings
  outputs/
    models/best_model.h5              Trained model
    detections/                       Detection CSVs
    auto_cleanup/                     Cleanup results and hard negatives
```

## Installation

```bash
git clone https://github.com/mo119m/primates-sound-detection.git
cd primates-sound-detection
pip install -r requirements.txt
```

YAMNet filter (auto-cleanup) additionally requires:
```bash
pip install tensorflow-hub resampy
```

## Dependencies

- TensorFlow 2.x
- librosa
- scikit-learn
- pandas, numpy, matplotlib
- soundfile
