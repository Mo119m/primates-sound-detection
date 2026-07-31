# Handoff — the V13 rebuild

Written 2026-07-31 so that a session on a different machine can pick this up
without the conversation that produced it. Branch: `v13-honest-labels`.

If you are an assistant reading this cold: everything below is established from
the data in this repo, and every claim names the script that reproduces it. The
short version is that the model was being trained on labels that were partly
wrong, and validated by a split that could not detect it.

---

## 1. What was found

### The validation split was measuring memorisation

`prepare_dataset` split train/validation **after** augmentation, at random.
With `AUGMENTATION_MULTIPLIER = 7` (9 for Colobus) and a 20 % validation share, a
clip keeps all its variants on one side only `0.8**7 = 21 %` of the time — so
about **79 % of source clips put a near-duplicate of a validation image into
training**. Colobus is worse: its 617 windows come from 172 recordings at a 1 s
hop, so neighbouring windows share half their audio before augmentation.

That is the gap between **98.12 % validation accuracy and 41.0 % field
precision**. The paper's whole accuracy table is measured this way.

Fixed: `train.grouped_split` keyed on the source recording
(`train.source_group`). Verified — 617 Colobus windows collapse to exactly 172
groups, 17 106 BirdNET segments to 3 222 recordings, and the simulation that
gives 79 % overlap under the old split gives 0 under the new one.
**The V12 numbers have not yet been re-measured under the honest split.** That
is job #1 below.

### One of the three cleanup filters was inverted

The README's Step 5 loop ("move flagged clips into a background folder, retrain,
repeat 3–5 times") had no ground truth: whatever the filters flagged became a
training negative and nobody listened.

Cross-matching `auto_flagged_fp` against the 6 189 reviewed detections localises
the damage exactly. The station-named subfolders — which were mined per station
and hand-checked — contain **no contradiction at all**. The two subfolders named
after *filters* do:

| Folder | clips | reachable by review | confirmed genuine calls |
|---|---|---|---|
| `mahal/` | 386 | 129 | **44 of 44 listened = 100 %** |
| `yamnet/` | 658 | 239 | 20 of 24 listened = 83 % |

A second listener labelled all 68 disputed clips by ear with nothing about the
model shown (`data/labels/disputed_68_labels.csv`): **64 of 68 are real calls**.
All four disagreements sit at confidence ≤ 0.73; of the 44 at confidence ≥ 0.90,
all 44 are real. Agreement with the original review is 94.1 %, which also says
the review itself is sound.

Mahalanobis flags whatever is far from the training distribution — and a loud,
unambiguous call is exactly that. YAMNet was already known to flag 51.8 % of
genuine calls, which is why `USE_YAMNET_FILTER` is off.

**657 clips in those two folders have never been labelled by anyone** and are
currently dropped from training for want of a label. At the measured base rate
roughly 118 of them are real calls. A listening batch is built and waiting:
`data/outputs/unlabelled_dumps/annotate.html`.

### The Colobus class does not work in the field, and the gate could not fix it

Of the 253 *C. guereza* detections the deployment produced, the user listened and
found **no genuine roar** — thunder and other low-frequency noise. 90.7 % fall
between 19:00 and 05:00; guereza roars at dawn. Median confidence 0.927.

`LOWFREQ_GATE_THRESHOLD` was 0.20, calibrated against `Colobus_confuser` clips
whose ratios top out at 0.092. Thunder is itself low-frequency (median 0.396,
p75 0.826) and overlaps the reference roars, so no setting separates them
cleanly. Recalibrated to **0.40**: reference roars kept 97.6 % → 93.4 %, field
detections kept 89.3 % → 49.4 % (`scripts/calibrate_colobus_gate.py`).

A second criterion was searched for and rejected: flatness, crest, envelope
variability and onset rate all score AUC 0.46–0.59; 1–8 Hz envelope modulation
reaches 0.715 but costs reference recall too fast to justify blind.

**There are zero field-verified Colobus positives, so Colobus field recall is
unmeasurable.** The paper should say so.

### Smaller things

- 20 clips from IPA19/IPA20 — the stations the config declares held out — were in
  training via the two filter dumps.
- `event_windows` average precision is inflated by row order: 24 % of detections
  share one integer value, and `effort_curve` sorts stably. Under random
  tie-breaking the honest figure is **0.9039**, not 0.9098. It still beats the
  four reported signals (0.8997). The manuscript passage is stale either way.

---

## 2. What was built

| Script | Does |
|---|---|
| `scripts/build_v13_dataset.py` | Assembles the manifest from human labels; attributes every clip to the stations it could have come from |
| `scripts/pack_v13_images.py` | Renders every clip to the 224×224 image the model eats (4.77 GB uint8) |
| `scripts/train_v13_loso.py` | Feature cache + leave-one-station-out training and scoring |
| `scripts/calibrate_colobus_gate.py` | Re-derives the gate against field negatives |
| `scripts/make_annotation_tool.py` | Builds a local listening page for labelling clips |
| `colab/v13_train.ipynb` | The same sweep on Colab (generated by `colab/make_notebook.py`) |

**The V13 training set** (`data/outputs/v13_manifest.csv`, 31 021 clips):

| Class | V12 | V13 |
|---|---|---|
| Cernic | 397 | **3 002** (2 535 confirmed field calls + 70 recovered + reference) |
| Background | 1 951 | **26 576** (3 654 confirmed field FPs + 17 101 BirdNET birds + reference) |
| Colobus_guereza | 789 | 789 |
| Colobus_confuser | 654 | 654 |

Every clip carries `possible_stations`. Eleven stations stamp a unique lat/lon
into each filename; **IPA1/2/4/6/7 recorded with GPS off and write identical
names**, and 50 files never locked GPS. Those 1 001 clips sit out every fold they
could belong to. Without this a leave-one-station-out number is decoration.

---

## 3. What to do on the GPU machine

### 3a. Check the GPU actually works first

```bash
python scripts/check_gpu.py
```

**This is not a formality.** The target machine has an **RTX 5070 Laptop**
(Blackwell, compute capability 12.0). TensorFlow's prebuilt wheels are compiled
for older architectures and reach Blackwell only through PTX JIT, which may be
slow or may not work at all; and TensorFlow dropped native Windows GPU support
after 2.10, so on Windows this needs **WSL2**. If `check_gpu.py` reports no GPU
or a JIT failure, do not spend hours debugging — fall back to
`colab/v13_train.ipynb`, which is known-good, and keep the GPU machine for the
re-detection pass, which is the part Colab genuinely cannot do.

### 3b. Rebuild the inputs (about 25 minutes)

```bash
python scripts/build_v13_dataset.py
python scripts/pack_v13_images.py
```

Needs the external drive mounted for the BirdNET negatives and the
coordinate→station map. On macOS that is `/Volumes/Gabon CNN`; edit `DRIVE` at
the top of `build_v13_dataset.py` for another mount point.

### 3c. The experiment

```bash
python scripts/train_v13_loso.py --folds all --epochs 15
```

Reads `v13_loso.csv`. **Read the matched-recall columns, not the deployment
threshold.** A model that fires less often removes false positives and loses
calls together, so "removed 68 % of false positives" means nothing without the
recall it was bought at — the degenerate model that answers Background to
everything removes 100 % of them, which is what the CPU smoke runs did.

### 3d. Re-detection, in stages

Re-detection is the expensive thing and the only thing that can measure recall.
Do not run it on all 3 014 recordings to find out whether the model improved.
Each 30-minute recording is 1 800 windows, which makes the unit costs:

| Scope | windows | CPU (9.6/s) | GPU (~300/s) |
|---|---|---|---|
| one recording | 1 800 | 3 min | 6 s |
| **one station** (192 files) | 345 600 | 10 h | **20 min** |
| all 16 stations | 5 400 000 | 156 h | ~5 h |

**One station is twenty minutes on a working GPU**, so there is no choice to make
between "run it all" and "run none of it". And because all 16 stations were
already reviewed detection by detection, a single station's re-run is scorable
immediately, with no new listening:

```bash
python scripts/run_detection_ipa.py --station IPA11ST \
    --model data/outputs/models/best_model_v13.h5 \
    --output data/outputs/detections_v13/IPA11ST
python scripts/compare_detection_to_review.py --station IPA11ST \
    --detections data/outputs/detections_v13/IPA11ST
```

The comparison reports, per station: confirmed calls still detected, confirmed
calls **lost**, false positives still made, false positives removed, and the
count of detections on ground the review never covered. Verified against V12's
own detections on five stations — it reproduces 100 % of the reviewed windows
with nothing spurious, so a real difference is the model's and not the tool's.

Suggested order, each stage gating the next:

1. **IPA11ST** — the hardest station (20.8 % precision, 42 calls / 160 false
   positives). If V13 cannot help here, it cannot help.
2. **IPA1ST, IPA17ST, IPA20ST** — 36.6 %, 70.0 %, 93.1 %, spanning the range.
   Four stations is ~80 GPU-minutes and covers the shape of the problem.
3. **Listen to the "new ground" detections** these produce. This is the only
   place absolute recall can be shown to improve — and the only place a new
   kind of false positive can hide, since the precision figures above say
   nothing about windows nobody has heard.
4. **The full run**, only once 1–3 hold up.

Two traps in `run_detection_ipa.py`, both now fixed, both worth knowing about
because older invocations carry them: it loaded models with
`keras.models.load_model`, which raises on the V11/V12 `FrequencyCoord` layer;
and its time filter defaulted **on** (05:30–10:30), while **81.5 % of the
reviewed detections fall outside that window**, so a default run reproduced less
than a fifth of the deployment and could not be compared with the review.

If full recall is still too expensive at the end, `scripts/recall_sample.py
plan|budget|score` bounds it from a few hours of exhaustively annotated audio.

---

## 4. Open, in priority order

1. **Re-measure V12 under the grouped split.** The paper's accuracy table
   (96.14 / 99.01 / 97.81 / 98.38 / 98.12) is currently unusable. Nothing else
   should be written up before this number exists.
2. **Run the 16-fold LOSO** (3c) — the actual V13 result.
3. **Label the 657** (`data/outputs/unlabelled_dumps/annotate.html`, ~1 h) and
   feed the CSV back; roughly 118 real calls are currently training as Background.
4. **Re-detection for recall** (3d).
5. **Manuscript**: the ranking passage numbers are stale (0.011 → 0.0152, CI,
   12/15 → 13/15, and `event_windows` at 0.9039); the Colobus claims need the
   field result stated; the accuracy table needs (1).

## 5. Where the data is

- Field recordings, 444 GB: `<drive>/Gabon raw acoustic data National Park/`,
  16 stations × 192 files
- BirdNET negatives, 17 106 clips: `<drive>/Gabon BirdNET segments Birds/`
- Reviewed detections: `reviews/*.csv` (6 189 rows) and
  `data/outputs/auto_cleanup/cleanup_vs_review.csv`
- Exported detection clips, 2.9 GB: `data/outputs/detected_clips/`
- Human labels: `data/labels/` — the one thing in `data/` that is versioned,
  because it is listening hours and cannot be regenerated
