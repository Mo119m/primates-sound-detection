# Primate Vocalization Detection

Automated detection of primate calls in long rainforest field recordings from
Makokou, Gabon. The production model (**V12**) is a four-class classifier
— *Cernic* (**Cercopithecus nictitans**, putty-nosed monkey), **Colobus
guereza**, a dedicated hard-negative *confuser* class, and *Background* — built
on a VGG19 backbone with a **frequency-position-aware** temporal-frequency CRNN
head (the `temporal_freqpos` head, 98.12% validation accuracy). A sliding-window
detector turns the classifier into a detector over continuous audio, with three
complementary false-positive controls for *Colobus*: a frequency-coordinate
(CoordConv) channel that lets the head reject high-frequency bird/insect sounds,
a **high-frequency nuisance augmentation** (V12) that breaks the model's
shortcut of keying on incidental high-frequency texture, and a **calibrated
low-frequency energy gate** (V12) that removes any residual high-frequency false
positive at detection time. A three-filter automatic cleanup pipeline then
recycles confirmed false positives as hard negatives for iterative retraining.

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
# Always use load_trained_model() — the V11/V12 model contains the custom
# FrequencyCoord layer, so raw tf.keras.models.load_model() fails with an
# unknown-layer error.
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
or the `PRIMATE_MODEL_POOLING` environment variable. **The production V12 model
uses `temporal_freqpos`** (the same head as V11; V12 adds high-frequency
nuisance augmentation and a calibrated low-frequency gate on top — see below);
the other heads are earlier iterations kept for provenance and ablation.

| Head | Config value | Description |
|---|---|---|
| GAP | `gap` | GlobalAveragePooling2D. Simple baseline (the code default). |
| Frequency-band | `freq_bands` | Split feature map into low/mid/high frequency bands, pool each separately (V6). |
| Temporal | `temporal` | Pool frequency axis, then Conv1D over time. Preserves *when* energy occurs (V7). |
| Temporal-frequency CRNN | `temporal_freq` | 4 frequency bands × per-band Conv1D → cross-band Conv1D → BiLSTM → GlobalMaxPool+GlobalAvgPool → Dense(512)→Dense(256). Preserves both *when* and *where* energy occurs (V8/V10). |
| **Temporal-frequency CRNN + FrequencyCoord** | `temporal_freqpos` | Extends `temporal_freq` with a `FrequencyCoord` CoordConv layer (see below) that stamps absolute frequency position onto the feature map before the band split. **Production head (V11/V12), 98.12% val accuracy.** |

### FrequencyCoord layer (`temporal_freqpos`, V11/V12)

VGG19's convolutions are translation-invariant along the frequency axis: a
rhythmic, harmonically structured call texture produces almost the same features
whether it sits low in the spectrogram (a *Colobus* roar) or high (a bird trill
or insect chorus). A head built on those features keys on the call *texture*
while discarding *where* in frequency it occurred — so high-frequency birds get
misclassified as low-frequency *Colobus*.

`FrequencyCoord` (in `src/model.py`) fixes this. It appends a normalized
frequency-coordinate channel to the `block4_conv4` feature map (0.0 at the
lowest mel row ≈ `FMIN`, 1.0 at the highest ≈ `FMAX`), and a
`Conv2D(128, 3×3) + BatchNorm + ReLU` fuses that coordinate with the texture
channels *before* the existing 4-band split → per-band Conv1D → cross-band
Conv1D → BiLSTM pipeline. Every downstream feature is then tagged with the
absolute frequency at which it occurs, so the model learns "this call texture
**at low frequency** = *Colobus*" and rejects the same texture higher up.

> **Loading caveat:** a V11/V12 model contains the custom `FrequencyCoord`
> layer, so it must be loaded with `model.load_trained_model(path)` (which passes
> the required `custom_objects`), **not** raw `tf.keras.models.load_model(path)`.

#### V12 validation accuracy (4-class)

| Class | V12 | V11 |
|---|---|---|
| Cernic | 96.14% | 93.82% |
| Colobus_guereza | 99.01% | 98.96% |
| Colobus_confuser | 97.81% | 97.38% |
| Background | 98.38% | 97.53% |
| **Overall** | **98.12%** | **97.29%** |

V12 keeps the V11 `temporal_freqpos` architecture and adds the high-frequency
nuisance augmentation (below). Breaking the high-frequency shortcut lifts every
class, with the largest gain on Cernic recall (+2.3 points) because the model no
longer confuses high-frequency Cernic texture with the confuser. (V10, the
`temporal_freq` head without frequency-position encoding, reached 96.14%.)

#### High-frequency nuisance augmentation (V12)

The curated *Colobus* reference clips carry incidental high-frequency bird and
insect energy that happens to correlate with the *Colobus* label, so the model
can take a shortcut — keying on high-frequency texture instead of the
low-frequency roar that is the true *Colobus* signature — and then mis-fire on
forest insects in the field. To break this spurious correlation, training adds
two extra variants per *Colobus* clip in which the mel-spectrogram band **above
1.5 kHz is replaced with the high band of a random background clip**, leaving the
low-frequency roar untouched (`augmentation.highfreq_nuisance`, gated by
`config.COLOBUS_HF_AUG_*`). Swapping varied high-frequency content across
examples decorrelates the high band from the label and forces the model to rely
on the invariant roar. It is applied **only** to *Colobus_guereza* — Cernic
calls genuinely occupy higher frequencies and must keep their high band intact.

### Four-class design and the confuser class

The model trains four classes — `Cernic`, `Colobus_guereza`, `Colobus_confuser`,
`Background` (the confuser class was introduced in V10 and is retained through
V12). The four putty-nosed call types (putty-nose, hacks, keks, pyows)
plus recovered field-confirmed calls are pooled into a single **Cernic** class.
The **`Colobus_confuser`** class is a *dedicated hard negative*: it collects the
forest sounds the detector repeatedly mis-fired as *Colobus*. Giving it its own
softmax output forces the model to learn the Colobus-vs-confuser boundary
explicitly instead of drowning a few hundred hard negatives in the generic
Background class. At detection time the confuser is folded into the Background
group (see `DETECTION_GROUPS` in `config.py`), so it never produces a detection.

### Low-frequency spectral-energy gate (V12, active)

A *Colobus* detection is kept only if the fraction of its spectral energy below
the 1500 Hz cutoff (within the `FMIN`–`FMAX` band) is at least
`LOWFREQ_GATE_THRESHOLD`; otherwise it is reclassified as Background. The gate
runs inside `detect_in_long_audio` after NMS (`LOWFREQ_GATE_ENABLED = True`) and
adds a `low_freq_ratio` column to every *Colobus* detection — a value that also
**doubles as a ranking signal**: sorting by it surfaces genuine low-frequency
roars (high ratio) and pushes insect false positives (near-zero ratio) to the
bottom of the review queue.

**Recalibration vs. the earlier gate.** An earlier version used a higher
threshold (0.40) with an ad-hoc `<1 kHz / full-spectrum` energy metric, and it
was too aggressive — genuine *C. guereza* clips with loud cicada noise could
fall below 0.40 and be wrongly rejected. V12 recalibrates the gate with the
**same metric the deployed code uses** (`detection.lowfreq_energy_ratio`) and a
threshold of **0.20**, chosen to sit below the reference-clip 5th percentile and
above the field false-positive 95th percentile:

| Group | Statistic | Low-freq ratio |
|---|---|---|
| Field false positives (insects) | max | 0.092 |
| Genuine *Colobus* clips | 5th percentile | 0.261 |
| **Gate threshold** | | **0.20** |

At 0.20 the gate cuts 100% of the measured false positives while keeping 97.6%
of true positives (a clean 0.11-wide margin between the two distributions). The
gate cannot create true positives — the model must still fire on a real roar
first (that is the job of the high-frequency nuisance augmentation above) — but
it reliably removes residual high-frequency false positives at detection time.
The gate code lives in `detection.lowfreq_energy_ratio` / `apply_lowfreq_gate`,
and `scripts/apply_lowfreq_gate.py` can apply it standalone to already-saved
clips.

## Repository Structure

```
src/                           Core library modules
scripts/                       Command-line entry points
data/                          Local drop-in workspace (put your audio here; git-ignored)
main_pipeline_notebooks/       Notebooks: main_local.ipynb (local) + run_in_colab.ipynb (Colab)
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
Data augmentation on mel-spectrograms to expand training data (7x multiplier;
*Colobus_guereza* gets `COLOBUS_HF_AUG_COUNT` extra high-frequency-randomized
variants on top).

| Function | Description |
|---|---|
| `augment_dataset()` | Augment entire dataset, producing X, y arrays and metadata |
| `augment_spectrogram()` | Augment one spectrogram (adds the V12 HF-nuisance variants for Colobus) |
| `add_background_noise()` | Mix spectrogram with background noise at random SNR |
| `highfreq_nuisance()` | Replace the band above the cutoff with random background high-frequency content (V12 Colobus shortcut-breaker) |
| `time_chop()` | Randomly crop along time axis |
| `freq_chop()` | Randomly crop along frequency axis |
| `translate()` | Shift spectrogram in frequency |

### model.py
VGG19-based transfer learning model with configurable pooling heads.

| Function | Description |
|---|---|
| `build_model()` | Build VGG19 + configurable pooling head and compile |
| `load_trained_model()` | Load a saved `.h5` model (passes `custom_objects` so V11/V12 `temporal_freqpos` models load) |
| `unfreeze_base_model()` | Unfreeze last N VGG19 blocks for fine-tuning |
| `get_callbacks()` | EarlyStopping, ModelCheckpoint, ReduceLROnPlateau |
| `FrequencyCoord` | Custom Keras layer (CoordConv): appends a normalized frequency-coordinate channel to the VGG19 feature map, making the `temporal_freqpos` head position-aware (V11/V12) |

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
| `annotate_lowfreq_ratio()` | Add the `low_freq_ratio` column to Colobus detections (ranking signal) |
| `apply_lowfreq_gate()` | Reclassify Colobus detections that fail the V12 low-frequency gate as Background |
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
| `main_local.ipynb` | **Local run** — drop data into `data/`, run end-to-end with zero path config (start here for local) |
| `run_in_colab.ipynb` | Full pipeline on Google Colab: mount Drive, train, detect (start here for Colab) |
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
| `MODEL_POOLING` | `gap` (code default; set to `temporal_freqpos` for the V11/V12 production model) | Pooling head: `gap`, `freq_bands`, `temporal`, `temporal_freq`, `temporal_freqpos` |
| `BATCH_SIZE` | 32 | Training batch size |
| `EPOCHS` | 50 | Max training epochs |
| `COLOBUS_HF_AUG_CLASS` | `Colobus_guereza` | Class that receives the V12 high-frequency nuisance augmentation |
| `COLOBUS_HF_CUTOFF_HZ` | 1500 Hz | Band above this is randomized in the HF augmentation |
| `COLOBUS_HF_AUG_COUNT` | 2 | Extra HF-randomized variants generated per Colobus clip |
| `DETECTION_CONFIDENCE_THRESHOLD` | 0.4 | Minimum confidence for detections |
| `LOWFREQ_GATE_ENABLED` | `True` | Apply the V12 low-frequency gate inside `detect_in_long_audio` |
| `LOWFREQ_GATE_CUTOFF` | 1500 Hz | Low-frequency gate boundary |
| `LOWFREQ_GATE_THRESHOLD` | 0.20 | Minimum low-frequency energy fraction to keep a Colobus detection (calibrated) |
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
   os.environ['PRIMATE_MODEL_POOLING'] = 'temporal_freqpos'   # production V11/V12 head
   ```
4. **`config.print_config_summary()`** should report
   `Classes: 4 (Cernic, Colobus_guereza, Colobus_confuser, Background)`.

That's it — the `drive.mount()` cell is the only Colab-specific step.

## Running Locally

### Easiest: drop-in `data/` folder + `main_local.ipynb`

The repo ships with a ready-made [`data/`](data/README.md) workspace whose
folders already match what the pipeline expects. **No path configuration, no
source edits.**

1. Clone the repo and `pip install -r requirements.txt`.
2. Copy your audio into the prepared folders (see [`data/README.md`](data/README.md)):
   - `data/species/<call-type>/` — labelled call clips (training positives)
   - `data/background/<...>/` — negative clips
   - `data/long_audio/` — continuous recordings to run detection on
3. Open **`main_pipeline_notebooks/main_local.ipynb`** and run the cells in order.

The first cell finds the repo root, points `PRIMATE_DATA_ROOT` at `data/`, and
selects the V11/V12 head automatically. Outputs land in `data/outputs/`. The
audio you drop in is git-ignored, so it never bloats your clone or gets
committed.

### Alternative: data outside the repo (`.env`)

If your data lives elsewhere, point the pipeline at it with environment
variables instead — every path is read from `src/config.py`, so **no source
code needs editing**.

```bash
cp .env.example .env
# edit .env: set PRIMATE_DATA_ROOT to your local data folder
set -a; source .env; set +a     # load the variables into your shell
```

`.env` sets two things that matter most:
- `PRIMATE_DATA_ROOT` — the folder holding `species/`, `background/`,
  `field_recordings/`, `outputs/` (see [Data Layout](#data-layout)).
- `PRIMATE_MODEL_POOLING=temporal_freqpos` — selects the production V11/V12 head
  (the code default is `gap`).

Verify with a one-shot check that packages import and the data folders are found:

```bash
python scripts/check_environment.py
```

Then use the same functions/scripts as the [Main Workflow](#main-workflow),
e.g. train with `train.run_complete_training_pipeline()` or detect a station
with `python scripts/run_detection_ipa.py --station IPA1ST`.

> Prefer not to use a `.env` file? Just export the variables inline:
> ```bash
> PRIMATE_DATA_ROOT=/path/to/data PRIMATE_MODEL_POOLING=temporal_freqpos \
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

To reproduce the published **V12** four-class model and field results:

1. **Environment.** `pip install -r requirements-frozen.txt` for exact version
   match (Python 3.10, TensorFlow 2.15, Google Colab). Or
   `pip install -r requirements.txt` for flexible versions.

2. **Select the production head.** The code default is `gap`; the published V12
   model uses the frequency-position-aware temporal-frequency CRNN. Set it via
   environment variable so no source edit is needed:
   ```bash
   export PRIMATE_MODEL_POOLING=temporal_freqpos
   ```
   The V12 high-frequency nuisance augmentation and low-frequency gate are
   on by default (`COLOBUS_HF_AUG_COUNT=2`, `LOWFREQ_GATE_ENABLED=True`), so no
   extra flags are needed.

3. **Point the pipeline at your data.** Lay out `species/`, `background/`, and
   the field recordings as in [Data Layout](#data-layout), then set
   `PRIMATE_DATA_ROOT` (and the other `PRIMATE_*` paths if they differ).
   Confirm with `config.print_config_summary()` — it should report
   `Classes: 4 (Cernic, Colobus_guereza, Colobus_confuser, Background)`.

4. **Train.** `train.run_complete_training_pipeline()` writes `best_model.h5`
   to `outputs/models/`. Two-stage training on the human-verified, label-audited
   clip pool reaches **98.12 %** validation accuracy (3471-clip stratified
   split), with near-zero confusion between the two primate classes.

5. **Detect.** Run detection per station, e.g.
   `python scripts/run_detection_ipa.py --station IPA1ST`, which exports one
   clip per detection that survives the low-frequency gate. Load V12 with
   `model.load_trained_model(...)` (it passes the `custom_objects` needed for
   the `FrequencyCoord` layer).

6. **(Optional) Clean up and iterate.** Run the three-filter auto-cleanup,
   fold confirmed false positives back into Background, and retrain (Steps 4–5
   in [Main Workflow](#main-workflow)).

Key parameters that fix the results — `SAMPLE_RATE`, `N_MELS`, `FMIN/FMAX`,
`WINDOW_SIZE/STRIDE`, `DETECTION_CONFIDENCE_THRESHOLD`, `COLOBUS_HF_AUG_COUNT`,
`COLOBUS_HF_CUTOFF_HZ`, `LOWFREQ_GATE_CUTOFF`, `LOWFREQ_GATE_THRESHOLD` — all
live in `src/config.py`.
