# Primate Vocalization Detection

Automated detection of primate calls in long rainforest field recordings from
Makokou, Gabon. The production model (**V12**) is a four-class classifier
— *Cernic* (**Cercopithecus nictitans**, putty-nosed monkey), **Colobus
guereza**, a dedicated hard-negative *confuser* class, and *Background* — built
on a VGG19 backbone with a **frequency-position-aware** temporal-frequency CRNN
head (`temporal_freqpos`, 98.12% validation accuracy).

> **Just want to run it?** → Follow [`SETUP.md`](SETUP.md) to install the
> environment (one time), download the pretrained model (see below), then open
> `main_pipeline_notebooks/main_local.ipynb` and run the cells top to bottom.
> See [`data/README.md`](data/README.md) for which folder each audio file goes in.
>
> **Prefer Google Colab (free GPU)?** → Open
> `main_pipeline_notebooks/run_in_colab.ipynb` in Colab.
>
> **Want to auto-clean false positives after detection?** → Open
> `auto_cleanup_local.ipynb` (local) or `auto_cleanup_false_positives.ipynb`
> (Colab) and run the cells.

## Pretrained Model

The production V12 model (`best_model_v12.h5`, ~80 MB) is required for
detection. It is not included in the repository due to file size.

**Download:** <!-- TODO: replace with actual link -->
> The pretrained model will be available at: [link to be added]
>
> After downloading, place the file at:
> ```
> data/outputs/models/best_model_v12.h5
> ```

If you prefer to train from scratch instead, run `main_local.ipynb` with
`FORCE_RETRAIN = True` in Step 3 (requires labelled training clips in
`data/species/`).

---

## Main Workflow

```
Step 1  Configure   →  Step 2  Train  →  Step 3  Detect  →  Step 4  Clean up  →  Step 5  Retrain
edit config.py         train.run_         detection.        auto_cleanup.          fold FPs into
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
paths (or set the `PRIMATE_*` environment variables).

```python
config.print_config_summary()
```

### Step 2 — Train the model

To reproduce the exact **V12** production model, use the two-stage script:

```bash
python scripts/train_v12.py
```

Two-stage schedule with the `temporal_freqpos` head: frozen VGG19 base (LR 1e-4)
→ fine-tune last two blocks (LR 1e-5). Saves `best_model_v12.h5` to
`data/outputs/models/`.

> The library one-liner `train.run_complete_training_pipeline()` runs a simpler
> single-stage schedule and uses the default pooling head (`gap`). To build the
> V12 head from it, set `PRIMATE_MODEL_POOLING=temporal_freqpos` first (the
> notebooks already do this).

### Step 3 — Detect in field recordings

```python
# Always use load_trained_model() — handles the custom FrequencyCoord layer.
model_obj = model.load_trained_model('data/outputs/models/best_model_v12.h5')

# One file:
detections = detection.detect_in_long_audio(model_obj, '/path/to/recording.wav')

# All files under LONG_AUDIO_ROOT:
all_detections = detection.process_all_long_audio_files(model_obj)
```

Or from the command line:
```bash
python scripts/run_detection_ipa.py --station IPA1ST
```

### Step 4 — Auto-cleanup false positives

Three filters (Mahalanobis OOD, YAMNet cross-check, temporal isolation) sort
detections into clean vs. suspicious, no manual listening needed.

```python
result = auto_cleanup.run_auto_cleanup(detection_dir='data/outputs/detections/IPA1ST')
result['clean_df']       # passed all filters
result['suspicious_df']  # flagged, with flag_reason column
```

### Step 5 — Retrain with hard negatives

Move flagged clips from `auto_cleanup/auto_flagged_fp/` into a background
folder, add it to `BACKGROUND_FOLDERS` in `config.py`, go back to Step 2.
Repeat 3–5 times.

---

## Getting Started

| Path | What to do |
|---|---|
| **[`SETUP.md`](SETUP.md)** | Step-by-step environment setup (Miniconda + pip) — Windows, macOS, Linux |
| **[`data/README.md`](data/README.md)** | Where each audio file goes (folder names, formats) |
| **`src/config.py`** | All paths, species definitions, and hyperparameters in one file |

### Install (short version)

```bash
conda create -n primates python=3.12 -y
conda activate primates
pip install -r requirements-frozen.txt
pip install jupyter
```

> Use **pip**, not `conda install tensorflow`. See [`SETUP.md`](SETUP.md)
> for the full walkthrough (including GPU setup).

---

## Repository Structure

```
src/                           Core library modules
scripts/                       Command-line entry points
data/                          Local drop-in workspace (put your audio here; git-ignored)
main_pipeline_notebooks/       Notebooks: local + Colab versions
presentation_notebooks/        Figures and slides for the paper
paper/                         MethodsX manuscript (LaTeX)
```

## Source Modules (`src/`)

### config.py
All paths, parameters, and species definitions in one place.

### data_loader.py

| Function | Description |
|---|---|
| `load_species_data()` | Load all species audio clips into a dictionary |
| `load_background_data()` | Load background noise clips from multiple folders |
| `load_audio_file()` | Load a single WAV file with padding/cropping to fixed length |
| `get_ipa_station_files()` | Get WAV files for an IPA station with optional time-of-day filtering |
| `get_long_audio_files()` | List all long audio files recursively |

### preprocessing.py

| Function | Description |
|---|---|
| `preprocess_audio()` | Audio waveform → mel-spectrogram → 224×224 RGB image |
| `extract_sliding_windows()` | Extract overlapping windows from long audio with timestamps |
| `audio_to_melspectrogram()` | Convert waveform to mel-spectrogram in dB scale |

### augmentation.py

| Function | Description |
|---|---|
| `augment_dataset()` | Augment entire dataset, producing X, y arrays and metadata |
| `augment_spectrogram()` | Augment one spectrogram (includes V12 HF-nuisance variants for Colobus) |
| `highfreq_nuisance()` | Replace band above cutoff with random background (V12 Colobus shortcut-breaker) |

### model.py

| Function | Description |
|---|---|
| `build_model()` | Build VGG19 + configurable pooling head and compile |
| `load_trained_model()` | Load a saved `.h5` model (passes `custom_objects` for FrequencyCoord) |
| `FrequencyCoord` | Custom Keras layer: appends frequency-coordinate channel to the feature map (V11/V12) |

### train.py

| Function | Description |
|---|---|
| `run_complete_training_pipeline()` | Full workflow: load → augment → two-stage train → evaluate |
| `prepare_dataset()` | Load audio, convert to spectrograms, augment, split train/val |

### detection.py

| Function | Description |
|---|---|
| `detect_in_long_audio()` | Full detection on one file: windows → predict → NMS → CSV |
| `process_all_long_audio_files()` | Detect across all files and aggregate results |
| `lowfreq_energy_ratio()` | Fraction of spectral energy below cutoff (V12 gate + ranking signal) |

### auto_cleanup.py

| Function | Description |
|---|---|
| `run_auto_cleanup()` | Orchestrate all three filters and save results |
| `filter_mahalanobis()` | Flag out-of-distribution detections |
| `filter_yamnet()` | Flag detections tagged as non-primate by YAMNet |
| `filter_temporal_isolation()` | Flag detections with no same-species neighbour within ±30 s |

## Scripts (`scripts/`)

| Script | Description |
|---|---|
| `check_environment.py` | Verify setup: packages, config paths, data folders |
| `run_detection_ipa.py` | Run detection on an IPA station |
| `run_auto_cleanup.py` | Run the three-filter cleanup from the command line |
| `train_v12.py` | Train the production V12 model |
| `apply_lowfreq_gate.py` | Apply the low-frequency gate to saved Colobus clips |
| `mine_field_negatives.py` | Mine false positives from dev-station recordings as hard negatives |
| `run_hard_negative_mining.py` | Extract medium-confidence predictions as candidate FPs |
| `tune_threshold.py` | Sweep confidence thresholds and report precision/recall |
| `summarize_review.py` | Aggregate the per-site manual-review CSVs (Kaleidoscope `MANUAL ID`) into per-station / per-species detection, confirmed-call, false-positive, and precision tallies for the paper |

## Notebooks (`main_pipeline_notebooks/`)

| Notebook | Description |
|---|---|
| **`main_local.ipynb`** | Local run — drop data into `data/`, run end-to-end, zero config |
| **`annotate_detections.ipynb`** | Local review UI — listen to each detection clip, label call / false-positive / unsure, and get the per-station tallies for the paper |
| **`auto_cleanup_local.ipynb`** | Local auto-cleanup — sort detections into clean vs. suspicious |
| `run_in_colab.ipynb` | Full pipeline on Google Colab (free GPU) |
| `auto_cleanup_false_positives.ipynb` | Auto-cleanup on Google Colab |

## Model Architecture (summary)

The V12 model uses VGG19 (ImageNet pretrained) with the `temporal_freqpos`
pooling head: a `FrequencyCoord` CoordConv layer stamps absolute frequency
position onto the feature map, then four frequency bands feed per-band Conv1D →
cross-band Conv1D → BiLSTM. This lets the model distinguish a low-frequency
*Colobus* roar from a high-frequency bird/insect trill with identical texture.

Three false-positive controls target *Colobus* specifically:
1. **High-frequency nuisance augmentation** — training swaps the high band of
   Colobus clips with random background, forcing the model to rely on the
   low-frequency roar (not incidental bird/insect energy).
2. **Confuser class** — a dedicated softmax output for the recurring forest sound
   that mimics Colobus, folded into Background at detection time.
3. **Low-frequency energy gate** — at detection time, a Colobus detection is kept
   only if most of its spectral energy sits below 1500 Hz (threshold 0.20).

| Class | Accuracy |
|---|---|
| Cernic | 96.14% |
| Colobus guereza | 99.01% |
| Colobus confuser | 97.81% |
| Background | 98.38% |
| **Overall** | **98.12%** |

See the [MethodsX paper](paper/) for full architectural details, design
rationale, and field validation results.

## Adapting to Your Own Species

The pipeline is **configuration-driven**. To use it for a different species or
site, no source code changes are needed:

1. Drop your labelled reference clips into `data/species/<your-class>/`
2. Drop negative clips into `data/background/<your-negatives>/`
3. Edit `SPECIES_FOLDERS` and `BACKGROUND_FOLDERS` in `src/config.py`
4. Run `main_local.ipynb` (or `run_in_colab.ipynb`) top to bottom

The number of output classes, class weighting, and detection grouping all follow
automatically from the configuration.

## License

MIT — see [LICENSE](LICENSE).
