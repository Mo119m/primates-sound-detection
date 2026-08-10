# Primate Vocalization Detection

Automated detection of primate calls in long rainforest field recordings from
Makokou, Gabon. The production model (**V12**) is a four-class classifier
— *Cernic* (**Cercopithecus nictitans**, putty-nosed monkey), **Colobus
guereza**, a dedicated hard-negative *confuser* class, and *Background* — built
on a VGG19 backbone with a **frequency-position-aware** temporal-frequency CRNN
head (`temporal_freqpos`). Measured leave-one-station-out on 6 189 human-reviewed
field detections, retraining on those verdicts raises precision from **0.70 to
0.93** at matched recall. Validation-set accuracy is deliberately not quoted —
see below.

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

The production V12 model (`best_model_v12.h5`, 142 MiB) is required for
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
detections into clean vs. suspicious.

```python
result = auto_cleanup.run_auto_cleanup(detection_dir='data/outputs/detections/IPA1ST')
result['clean_df']       # passed all filters
result['suspicious_df']  # flagged, with flag_reason column
```

**Two of the three filters do not work, measured against the 6 189 human
verdicts in `data/outputs/auto_cleanup/cleanup_vs_review.csv`:**

| Filter | What it flags | Verdict |
|---|---|---|
| YAMNet | nothing — `USE_YAMNET_FILTER = False` | disabled; it flags 51.8 % of real calls |
| Mahalanobis | 310 rows, 26.1 % of them genuine calls | ROC area 0.515 = chance. It preferentially flags *loud, unambiguous* calls, which are exactly what sits far from the training distribution. A blind re-listen of 44 clips it exported as false positives judged **all 44 genuine**. |
| Temporal isolation | 530 rows, 7.5 % genuine | the only one that helps — and it is being wasted, see below |

`flag_isolated` is exactly `n_neighbours == 0`, i.e. a count that ranges 0–58
collapsed into one bit. Used as a graded value *after* a time gate it is the
strongest signal in the file (ROC area 0.880, same direction at 16 of 16
stations). Re-derive both with `scripts/calibrate_cleanup.py`, which fits
leave-one-station-out so the threshold is never chosen on the station it is
scored against:

```
no gate                                  precision 0.410, recall 1.000
time gate 05:00–19:00                    precision 0.701, recall 0.987
+ n_neighbours (threshold fitted LOSO)   precision 0.869, recall 0.898
```

### Step 5 — Retrain with hard negatives

> **Do not do this the way it used to be described here.** The old instruction
> was: move everything in `auto_cleanup/auto_flagged_fp/` into a background
> folder, retrain, repeat 3–5 times. That loop has no ground truth — whatever
> the filters flagged became a training negative and nobody listened. It fed the
> model's most confident *genuine* calls back as Background, including 20 clips
> from IPA19/IPA20, the two stations the configuration holds out.
> `config.BACKGROUND_EXCLUDE` now blocks the two poisoned subfolders
> (`mahal/`, `yamnet/`) at load time; read the comment there before touching it.

The station-named subfolders under `auto_flagged_fp/` **are** sound: they were
mined per station and hand-checked, and cross-matching them against the review
turns up no contradiction. They load as Background automatically (3 300 clips).

To add genuinely new hard negatives, mine them against human labels rather than
against a filter — `scripts/mine_field_negatives.py` — and check any new pool
against `data/outputs/auto_cleanup/cleanup_vs_review.csv` before training on it.

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
tools/                         Standalone browser tools (see Audio Labeler below)
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
| `evaluate_cleanup.py` | Score the auto-cleanup against those manual labels (ground truth): false positives removed, genuine calls retained, precision before vs. after |

---

## Audio Labeler (`tools/labeler.html`)

**A single HTML file for labelling a folder of audio by ear. No install, no
server, no network — open it in a browser and go.**

It is not specific to this project or to primates: point it at any folder of
audio, define whatever categories your question needs, and export a CSV.

```
open tools/labeler.html
```

1. **Choose a folder** — subfolders included; WAV, MP3, OGG, FLAC and M4A are read.
2. **Define the categories** — type your own, or start from a preset. Keys
   <kbd>1</kbd>–<kbd>9</kbd> map to them in order.
3. **Label** — <kbd>space</kbd> replays, arrows move, it advances automatically.
4. **Export CSV.**

Spectrograms are computed in the page (small radix-2 FFT), so any folder works
with nothing prepared in advance. Nothing is uploaded; the files never leave the
machine, and answers persist per folder in the browser, so a half-finished pass
survives a closed tab.

Three defaults exist because of what went wrong without them:

- **Order is shuffled** (seeded, so it is stable across sessions). Clips cut from
  one recording arrive adjacent, and labelling a run of them invites a run of
  identical judgements.
- **Filenames can be hidden.** Ours carry the model's confidence, and a listener
  who knows the model was certain labels differently — which defeats the purpose
  of an independent opinion.
- **Gain up to 15×.** Field recordings are quiet, and a faint call at the edge of
  an analysis window is easy to miss at 1×.

### Sending a batch to someone else

A `file://` path resolves on the *recipient's* machine, so sending a colleague a
path sends them a path to files they do not have.
`scripts/make_annotation_tool.py --standalone` builds one self-contained HTML
with the audio and spectrograms embedded — put it in Drive, they download and
double-click it, and it works offline.

```bash
python scripts/make_annotation_tool.py --clips <folder> --standalone \
    --labels "Thunderstorm,Wood cut,Rain,Bird or insect,Real call,Unknown" \
    --parts 3            # 3 emailable files instead of one large one
```

Audio is re-encoded to 22.05 kHz Ogg Vorbis for this (the analysis band ends at
8 kHz, so nothing audible to the task is lost), which takes a 250-clip page from
hundreds of megabytes to about 18 MB.

---

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
   only if most of its spectral energy sits below 1500 Hz. The threshold is
   **0.40** (`config.LOWFREQ_GATE_THRESHOLD`); the 0.20 used in the deployment
   was calibrated against confuser clips whose ratios top out at 0.09, and it
   removed **none** of the 253 field detections, whose minimum is 0.2007.

### Performance

**There is no validation-accuracy table here any more, and that is deliberate.**
Earlier versions of this README quoted 98.12 % overall with a per-class
breakdown. Those numbers came from a split taken *after* augmentation: with a
multiplier of seven, a source clip keeps all its variants on one side only
`0.8^7 = 21 %` of the time, so roughly 79 % of source clips put a near-duplicate
across the split. They measured memorisation, and the distance between them and
the 41.0 % precision the model actually achieved in the field is the size of the
problem.

What replaces them is a field measurement on stations the model never saw
(`scripts/train_v13_loso.py`, `data/outputs/v13_loso.csv`), macro-averaged over
all 16 folds at matched 95 % recall:

| | Deployed (V12) | Retrained | |
|---|---|---|---|
| Precision | 0.695 | **0.934** | at 95.3 % of the calls retained |
| False positives removed | — | **82.1 %** | weakest station IPA7ST at 0.759 |

Two free post-processing steps land before the model does any work at all
(`scripts/calibrate_time_gate.py`, `scripts/calibrate_cleanup.py`):

| Stage | Precision | Recall |
|---|---|---|
| ungated | 0.410 | 1.000 |
| + time gate 05:00–19:00 | 0.701 | 0.987 |
| + graded `n_neighbours`, LOSO-fitted | 0.869 | 0.898 |

*Colobus guereza* is reported as a **negative result**: 253 field detections, all
listened to, none genuine. See the [MethodsX paper](paper/) for the four
independent searches behind that and for architectural details.

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
