# Primate Vocalization Detection

Automated detection of primate calls in long rainforest field recordings from
Makokou, Gabon. The production model (**V10**) is a four-class classifier
— *Cernic* (**Cercopithecus nictitans**, putty-nosed monkey), **Colobus
guereza**, a dedicated hard-negative *confuser* class, and *Background* — built
on a VGG19 backbone with a temporal-frequency CRNN head. A sliding-window
detector turns the classifier into a detector over continuous audio, with two
complementary false-positive controls: a low-frequency spectral-energy gate on
*Colobus* detections, and a three-filter automatic cleanup pipeline whose
confirmed false positives are recycled as hard negatives for iterative
retraining.

> **Reproducing the published results?** Jump to [Reproducibility](#reproducibility).

## Main Workflow

The pipeline has 5 stages. Each one below shows **exactly which function to call**.
Easiest path: open `main_pipeline_notebooks/run_in_colab.ipynb` and run the cells,
which call these same functions for you.

```
Step 1  Configure   ->  Step 2  Train  ->  Step 3  Detect  ->  Step 4  Clean up  ->  Step 5  Retrain
edit config.py          train.run_         detection.        auto_cleanup.          fold FPs into
                        complete_          process_all_      run_auto_cleanup()     Background, go to
                        training_          long_audio_                              Step 2 (iterate)
                        pipeline()         files()
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
augmentation -> two-stage train -> evaluate). It saves `best_model.h5` to `outputs/models/`.

```python
trained_model = train.run_complete_training_pipeline()
```

Training uses a two-stage schedule:
1. **Frozen base** — VGG19 convolutional layers frozen, only the pooling head and dense classifier train (LR = 1e-4).
2. **Fine-tune** — last VGG19 block(s) unfrozen for end-to-end fine-tuning (LR = 1e-5).

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

**Label audit**: after several self-training rounds, review the accumulated
hard negatives for label pollution — real target species calls that were
wrongly mined as false positives. Confirmed true positives should be moved to
the corresponding `species/` folder (e.g. `species/CERNIC field_confirmed`).

## Model Architecture

The model uses VGG19 (pretrained on ImageNet) as a feature extractor with a
configurable pooling head. Set the head via the `MODEL_POOLING` config option
or the `PRIMATE_MODEL_POOLING` environment variable. **The production V10 model
uses `temporal_freq`**; the other heads are earlier iterations kept for
provenance and ablation.

| Head | Config value | Description |
|---|---|---|
| GAP | `gap` | GlobalAveragePooling2D. Simple baseline (the code default). |
| Frequency-band | `freq_bands` | Split feature map into low/mid/high frequency bands, pool each separately (V6). |
| Temporal | `temporal` | Pool frequency axis, then Conv1D over time. Preserves *when* energy occurs (V7). |
| **Temporal-frequency CRNN** | `temporal_freq` | 4 frequency bands × per-band Conv1D → cross-band Conv1D → BiLSTM → GlobalMaxPool+GlobalAvgPool → Dense(512)→Dense(256). Preserves both *when* and *where* energy occurs. **Production head (V10), ~12.6M params.** |

### Four-class design and the confuser class

V10 trains four classes — `Cernic`, `Colobus_guereza`, `Colobus_confuser`,
`Background`. The four putty-nosed call types (putty-nose, hacks, keks, pyows)
plus recovered field-confirmed calls are pooled into a single **Cernic** class.
The **`Colobus_confuser`** class is a *dedicated hard negative*: it collects the
forest sounds the detector repeatedly mis-fired as *Colobus*. Giving it its own
softmax output forces the model to learn the Colobus-vs-confuser boundary
explicitly instead of drowning a few hundred hard negatives in the generic
Background class. At detection time the confuser is folded into the Background
group (see `DETECTION_GROUPS` in `config.py`), so it never produces a detection.

### Low-frequency spectral-energy gate

A complementary, post-hoc gate is applied to *Colobus* detections only. For each
detected clip it computes the fraction of spectral energy below a 1500 Hz cutoff
(within the 20–8000 Hz band, same STFT parameters as the pipeline) and rejects
the detection if that fraction falls below a threshold (default 0.40). Genuine
*C. guereza* roars are overwhelmingly low-frequency, whereas the most common
out-of-distribution false positives (insects, cicadas) are high-frequency, so
the gate removes them without touching real calls. Because it runs on saved
clips, it requires no retraining.

## Repository Structure

```
src/                           Core library modules
scripts/                       Command-line entry points
main_pipeline_notebooks/       Colab notebooks for training, detection, and cleanup
presentation_notebooks/        Figures and slides generation
paper/                         MethodsX manuscript (LaTeX)
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
VGG19-based transfer learning model with configurable pooling heads.

| Function | Description |
|---|---|
| `build_model()` | Build VGG19 + configurable pooling head and compile |
| `load_trained_model()` | Load a saved `.h5` model |
| `unfreeze_base_model()` | Unfreeze last N VGG19 blocks for fine-tuning |
| `get_callbacks()` | EarlyStopping, ModelCheckpoint, ReduceLROnPlateau |

### train.py
End-to-end training pipeline with two-stage schedule.

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
| `get_detection_groups()` | Build detection labels and the class indices feeding each group |
| `group_probabilities()` | Sum softmax scores across detection groups before thresholding |
| `sweep_thresholds()` | Apply multiple confidence thresholds to pre-computed predictions |
| `apply_nms()` | Non-Maximum Suppression to remove overlapping detections |
| `lowfreq_energy_ratio()` | Fraction of a clip's spectral energy below the low-frequency cutoff |
| `apply_lowfreq_gate()` | Drop Colobus detections that fail the low-frequency gate |
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
| `check_environment.py` | Verify local setup: packages import, config paths resolve, data folders exist |
| `run_detection_ipa.py` | Run detection on an IPA field recording station. Args: `--station`, `--model`, `--threshold`, `--no-time-filter` |
| `run_auto_cleanup.py` | Run the three-filter false-positive cleanup. Args: `--detection-dir`, `--model`, `--percentile`, `--isolation-window` |
| `apply_lowfreq_gate.py` | Apply the low-frequency spectral-energy gate to saved Colobus detection clips. Args: `--clip-root`, `--station`, `--cutoff`, `--threshold`, `--move-rejected` |
| `run_hard_negative_mining.py` | Extract medium-confidence predictions as candidate false positives for retraining |
| `mine_field_negatives.py` | Mine confirmed false positives from dev-station field recordings as distribution-matched hard negatives |
| `filter_recordings_by_time.py` | Copy only recordings within a time-of-day window (pre-upload filter) |
| `analyze_detections.py` | Per-species detection analysis and threshold suggestion helpers |
| `train_v7_temporal.py` | Train with the temporal (V7) pooling head |
| `train_v8_temporal_freq.py` | Train with the temporal-frequency CRNN (V8) pooling head |
| `tune_threshold.py` | Sweep detection confidence thresholds and report precision/recall |
| `visualize_fp_vs_tp.py` | Side-by-side spectrogram comparison of true vs. false positives |
| `fetch_colobus_library.py` | Download Colobus guereza reference calls from online sound libraries |
| `acoustic_feature_separation.py` | Acoustic feature analysis for separating target vs. confuser species |

## Notebooks

### Main pipeline (`main_pipeline_notebooks/`)

| Notebook | Description |
|---|---|
| `run_in_colab.ipynb` | Full pipeline: setup, train, detect (start here) |
| `main_pipeline_updated.ipynb` | Detailed step-by-step training and evaluation |
| `auto_cleanup_false_positives.ipynb` | Run auto-cleanup interactively with visualization |

### Presentation (`presentation_notebooks/`)

| Notebook / Script | Description |
|---|---|
| `01_data_overview.ipynb` | Dataset statistics and species distribution figures |
| `02_model_results.ipynb` | Training curves, confusion matrices, per-class metrics |
| `03_detection_analysis.ipynb` | Field detection results and temporal patterns |
| `make_architecture_figure.py` | Generate model architecture diagram |
| `make_augmentation_figure.py` | Generate augmentation examples figure |
| `make_pipeline_figures.py` | Generate pipeline overview figures |

## Configuration

All parameters live in `src/config.py`. Key settings:

| Parameter | Default | Description |
|---|---|---|
| `SAMPLE_RATE` | 44100 Hz | Audio sample rate |
| `CLIP_DURATION` | 2.0 s | Training clip length |
| `WINDOW_SIZE` / `WINDOW_STRIDE` | 2.0 / 1.0 s | Detection sliding window |
| `N_MELS` | 128 | Mel-spectrogram frequency bins |
| `FMIN` / `FMAX` | 20 / 8000 Hz | Frequency range |
| `MODEL_POOLING` | `gap` | Pooling head: `gap`, `freq_bands`, `temporal`, `temporal_freq` |
| `BATCH_SIZE` | 32 | Training batch size |
| `EPOCHS` | 50 | Max training epochs |
| `DETECTION_CONFIDENCE_THRESHOLD` | 0.4 | Minimum confidence for detections |
| `LOWFREQ_GATE_CUTOFF` | 1500 Hz | Low-frequency gate boundary |
| `LOWFREQ_GATE_THRESHOLD` | 0.40 | Minimum low-frequency energy fraction to keep a Colobus detection |
| `TIME_FILTER_START` / `END` | 05:30 / 10:30 | Field recording time window |

Override data paths via environment variables: `PRIMATE_DATA_ROOT`, `PRIMATE_AUDIO_ROOT`, `PRIMATE_LONG_AUDIO_ROOT`, `PRIMATE_IPA_ROOT`, `PRIMATE_OUTPUT_ROOT`.

## Data Layout

```
primates-data/                          (PRIMATE_DATA_ROOT)
  species/
    CERNIC putty-nose 2s/               Putty-nosed monkey putty-nose calls (2 s clips)
    CERNIC hacks/                       Putty-nosed monkey hack calls
    CERNIC keks/                        Putty-nosed monkey kek calls
    CERNIC pyows/                       Putty-nosed monkey pyow calls
    CERNIC field_confirmed/             Real Cernic calls recovered from label audit
    Colobus guereza 2s windows/         Colobus guereza calls
    Colobus_confuser/                   Hard-negative confuser class (mined Colobus FPs)
  background/
    background noise Clips 5sec/        Environmental noise
    Cercocebus torquatus Clips 5s/      Non-target species
    wrong classified/                   Misclassified examples
    Pan troglodytes Clips 5sec/         Non-target species
    field_fp_negatives/                 Distribution-matched hard negatives from dev stations
  field_recordings/
    IPA1ST/YYYYMMDD/*.wav               IPA station field recordings
  outputs/
    models/best_model.h5                Trained model
    detections/                         Detection CSVs
    auto_cleanup/                       Cleanup results and hard negatives
      auto_flagged_fp/                  Accumulated hard negatives from self-training loop
```

All four Cernic call types (putty-nose, hacks, keks, pyows) plus recovered
field-confirmed calls are merged into a single **Cernic** class. The
`Colobus_confuser` folder holds the mined hard negatives that train the
confuser class. The `auto_flagged_fp` folder accumulates false positives across
self-training iterations; `scan_audio_files()` walks it recursively, so all
subfolders are loaded as Background during training.

## Installation

```bash
git clone https://github.com/mo119m/primates-sound-detection.git
cd primates-sound-detection
pip install -r requirements.txt
```

For exact reproducibility (Python 3.10 / Google Colab environment matching the
published results with TensorFlow 2.15), use the frozen requirements instead:

```bash
pip install -r requirements-frozen.txt
```

All dependencies including `tensorflow-hub` and `resampy` (needed for the YAMNet
auto-cleanup filter) are included in both requirements files.

## Running on Colab

The easiest path. Open `main_pipeline_notebooks/run_in_colab.ipynb` in Google
Colab and run the cells in order — it walks through every step with checkpoints.

1. **Runtime → Change runtime type → T4 GPU.**
2. **Put your data on Google Drive** at `My Drive/primates-data/` with the
   subfolders `species/`, `background/`, `long_audio/` (and `field_recordings/`
   for detection) — see [Data Layout](#data-layout).
3. **Run the notebook cells.** They mount Drive, clone this repo, install
   `requirements.txt`, set the environment, and run the pipeline. The two
   environment variables the notebook sets for you are:
   ```python
   os.environ['PRIMATE_DATA_ROOT']     = '/content/drive/MyDrive/primates-data'
   os.environ['PRIMATE_MODEL_POOLING'] = 'temporal_freq'   # production V10 head
   ```
4. **`config.print_config_summary()`** should report
   `Classes: 4 (Cernic, Colobus_guereza, Colobus_confuser, Background)`.

That's it — the `drive.mount()` cell is the only Colab-specific step.

## Running Locally

The pipeline runs the same locally as in Colab — **no source code needs to be
edited or commented out**. Every path is read from environment variables
(see `src/config.py`), and the only Colab-specific code is the `drive.mount()`
cell in the notebooks, which you simply skip when running locally.

**1. Configure your paths.** Copy the template and edit it:

```bash
cp .env.example .env
# edit .env: set PRIMATE_DATA_ROOT to your local data folder
set -a; source .env; set +a     # load the variables into your shell
```

`.env` sets two things that matter most:
- `PRIMATE_DATA_ROOT` — the folder holding `species/`, `background/`,
  `field_recordings/`, `outputs/` (see [Data Layout](#data-layout)).
- `PRIMATE_MODEL_POOLING=temporal_freq` — selects the production V10 head
  (the code default is `gap`).

**2. Verify the setup.** A one-shot check that packages import and the data
folders are found:

```bash
python scripts/check_environment.py
```

It prints `[ OK ]` / `[WARN]` / `[FAIL]` per item and exits non-zero if anything
required is missing.

**3. Run.** Use the same functions/scripts as the [Main Workflow](#main-workflow),
e.g. train with `train.run_complete_training_pipeline()` or detect a station
with `python scripts/run_detection_ipa.py --station IPA1ST`. Do **not** run the
`drive.mount(...)` notebook cell — that is the single Colab-only step.

> Prefer not to use a `.env` file? Just export the variables inline:
> ```bash
> PRIMATE_DATA_ROOT=/path/to/data PRIMATE_MODEL_POOLING=temporal_freq \
>     python scripts/run_detection_ipa.py --station IPA1ST
> ```

## Dependencies

- TensorFlow 2.x
- librosa
- scikit-learn
- pandas, numpy, matplotlib
- soundfile
- tensorflow-hub, resampy (YAMNet auto-cleanup filter)

## Reproducibility

To reproduce the published **V10** four-class model and field results:

1. **Environment.** `pip install -r requirements-frozen.txt` for exact version
   match (Python 3.10, TensorFlow 2.15, Google Colab). Or
   `pip install -r requirements.txt` for flexible versions.

2. **Select the production head.** The code default is `gap`; the published
   model uses the temporal-frequency CRNN. Set it via environment variable so
   no source edit is needed:
   ```bash
   export PRIMATE_MODEL_POOLING=temporal_freq
   ```

3. **Point the pipeline at your data.** Lay out `species/`, `background/`, and
   the field recordings as in [Data Layout](#data-layout), then set
   `PRIMATE_DATA_ROOT` (and the other `PRIMATE_*` paths if they differ).
   Confirm with `config.print_config_summary()` — it should report
   `Classes: 4 (Cernic, Colobus_guereza, Colobus_confuser, Background)`.

4. **Train.** `train.run_complete_training_pipeline()` writes `best_model.h5`
   to `outputs/models/`. Two-stage training on the human-verified, label-audited
   clip pool reaches **96.14 %** validation accuracy (3471-clip stratified
   split), with near-zero confusion between the two primate classes.

5. **Detect.** Run detection per station, e.g.
   `python scripts/run_detection_ipa.py --station IPA1ST`, which exports one
   clip per detection.

6. **Gate Colobus detections.** Apply the low-frequency gate to the saved clips:
   ```bash
   python scripts/apply_lowfreq_gate.py --clip-root <OUTPUT_ROOT>/detection_clips_model_v10
   ```

7. **(Optional) Clean up and iterate.** Run the three-filter auto-cleanup,
   fold confirmed false positives back into Background, and retrain (Steps 4–5
   in [Main Workflow](#main-workflow)).

Key parameters that fix the results — `SAMPLE_RATE`, `N_MELS`, `FMIN/FMAX`,
`WINDOW_SIZE/STRIDE`, `DETECTION_CONFIDENCE_THRESHOLD`, `LOWFREQ_GATE_CUTOFF`,
`LOWFREQ_GATE_THRESHOLD` — all live in `src/config.py`.
