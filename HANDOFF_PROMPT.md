# Primate bioacoustics pipeline — state, locations, and the open question

Copy everything below to another agent.

---

## What this project is

A detector for primate vocalisations in passive acoustic monitoring audio from
16 stations in Ivindo National Park, Gabon. Written up as a MethodsX paper. The
target species is *Cercopithecus nictitans* (putty-nosed monkey), internally
called `Cernic`. Three other classes exist: `Colobus_guereza`, `Colobus_confuser`
(a dedicated hard-negative class), and `C_pogonias` (a congener the species
expert identified by ear as the source of many false positives).

MethodsX has no major-revision path — a reviewer who asks for another experiment
causes rejection, not revision — so everything must be finished before
submission.

## Machine and environment

- Windows 11. Python: `C:/Users/Fudap/miniconda3/envs/primates/python.exe`
- TensorFlow 2.20 has **no GPU on native Windows**. Everything runs on CPU at
  about 33 images/s. A 16-fold sweep is 7–13 hours depending on dataset size.
  The machine has an RTX 5070 that TensorFlow cannot see.
- 31.5 GB RAM. A sweep holds ~18 GB, so the machine is unusable for anything
  else while one runs.

## Where the data is

**External drive D: (444 GB, cannot be moved)**

| Path | Size | What |
|---|---|---|
| `D:/Gabon raw acoustic data National Park/IPA*/` | 444 GB | The deployment. 3 014 WAV files, 16 stations (IPA1ST…IPA20ST), 44.1 kHz mono, ~30 min each, Feb 2021. 2 files unreadable. |
| `D:/Primates training data/` | 309 MB | Library reference audio by species. Colobus is under `Colobus gereza_Black and white colobus/` (79 files, `bwcolob*`). |
| `D:/Gabon BirdNET segments Birds/` | 8.7 GB | 17 106 clips, BirdNET's own detections on field audio, foldered by eBird species code. **Provenance: predates the current work, present in the remote git branch. Nobody has listened to them individually.** |

**Repository `C:/Users/Fudap/primates-sound-detection` (branch `v13-honest-labels`)**

| Path | What |
|---|---|
| `data/species/` | Curated reference clips per class |
| `data/species/Colobus guereza pulses/` | 665 single roar pulses, 0.25–1.60 s, cut by `scripts/segment_roar_pulses.py` |
| `data/species/C_pogonias/` | 150 library clips |
| `data/species/Colobus_confuser/` | 907 hard negatives |
| `data/background/` | Curated negatives |
| `data/background/random_forest/` | 19 573 random deployment windows. **WITHDRAWN from the config — see open question 1** |
| `data/outputs/v13_manifest.csv` | 31 047 rows: path, label, station, possible_stations, source |
| `data/outputs/v13_index.csv` + `v13_images.npy` | One 224×224×3 image per manifest row |
| `data/outputs/v13_features.npy` | 25 GB cache of frozen VGG19 `block4_conv4` activations |
| `data/outputs/auto_cleanup/cleanup_vs_review.csv` | **The ground truth.** 6 189 human verdicts on deployed detections: 2 535 `call`, 3 654 `false_positive` |
| `data/outputs/v13_heads/` | 16 trained heads, one per fold |
| `data/outputs/v13_loso_*.csv` | Results, one file per configuration |
| `paper/methodsx_manuscript.tex` | The manuscript |
| `SESSION_2026-08-03.md` | What was found and fixed, including mistakes |

Set `PRIMATE_IPA_ROOT='D:/Gabon raw acoustic data National Park'` for any script
that reads field audio.

## The pipeline

```
reference clips + field detections
  -> scripts/build_v13_dataset.py    manifest with per-clip station attribution
  -> scripts/pack_v13_images.py      one mel-spectrogram image per clip
  -> scripts/train_v13_loso.py       frozen VGG19 -> CRNN head, 16 folds
  -> scripts/assemble_fold_model.py  weld head + trunk into a deployable .h5
  -> scripts/run_detection_ipa.py    sliding-window detection over raw audio
```

Model: VGG19 (ImageNet, **frozen**) tapped at `block4_conv4` → a frequency
coordinate channel (CoordConv) → 4 frequency bands each with its own Conv1D →
cross-band Conv1D → BiLSTM → dense → 5-way softmax. Detection collapses the
softmax onto `config.DETECTION_GROUPS` before the argmax.

Validation: leave-one-station-out, 16 folds. A clip is withheld from a fold
whenever the held-out station appears in its `possible_stations`. Verified by an
independent reimplementation: zero train/eval overlap in all 16 folds, and the
16 evaluate masks partition the 6 189 reviewed windows exactly.

## Current results

Macro over 16 folds, inside the deployed 05:00–19:00 window, threshold fitted
off the held-out station and applied to it, deployed grouped-argmax on:

| configuration | precision | calls kept | F1 | file |
|---|---|---|---|---|
| deployed baseline (V12) | 0.6953 | — | — | — |
| 4 classes | 0.9535 | 0.8596 | 0.9041 | `v13_loso_final.csv` |
| + C_pogonias as its own target | 0.9556 | 0.8687 | 0.9100 | `v13_loso_2target.csv` |
| + Colobus trained on pulses | 0.9420 | 0.8666 | 0.9027 | `v13_loso_pulses.csv` |
| + 19 573 random background | 0.9431 | 0.9013 | 0.9217 | `v13_loso_randombg.csv` |

The manuscript reports `v13_loso_final.csv`.

## What is verified, and what is not

**Verified by independent audit or direct measurement:**
- No train/eval overlap in any fold; the masks agree with an independent
  reimplementation row for row
- The 18 165 clips with no station attribution are all ≥51.4 km from the nearest
  station and none is dated to the 2021-02 deployment
- Image row *i* is manifest row *i*; `ok` is true for every row
- No clip labelled `call` in the review appears in the Background class
- Station attribution of every random-background clip matches the drive
- 167 unit tests pass

**Not verified:**
- **65 % of the Background class (17 101 BirdNET clips) has never been listened
  to.** If real *C. nictitans* calls are in there, the model is trained to reject
  the calls it should find.
- The 19 573 random windows were labelled Background by a model whose recall is
  unknown. Circular. Withdrawn pending a listening check.
- No recall figure exists for anything. Every reported number is precision on
  windows the deployed model already fired on.

## The problem that motivated the last week

The head re-ranks the deployed detector's candidates well (0.6953 → 0.9535). It
**cannot scan raw audio**. Assembled into a whole model and run over one
station's daytime recordings it produced 1 425 detections against the
deployment's 131 in the same window, and every clip checked by ear was wrong, at
confidences up to 0.988.

The diagnosis, measured rather than guessed: 93 % of the Background class is
derived from detector output (the deployed model's false positives, reviewed
detections, or BirdNET's own detections). The class describes *what a detector
already reacted to*, not *what the forest contains*, which is enough to re-rank a
candidate list and not enough to scan a recording.

Adding 19 573 uniformly random windows made scanning **worse** (1 425 detections,
up from 1 154). The reason is `sklearn` `balanced` class weights: they give every
class an equal share of the loss, so adding negatives to an already-dominant
class does not raise its influence — it raises every other class's per-sample
weight. Measured: Cernic's per-sample weight went 1.98 → 3.17, C_pogonias 37 →
62.6, when 19 573 Background clips were added.

## Two reference papers whose methods are partly borrowed

1. **Sun et al. 2022, Ecological Indicators 145:109621** — the paper this
   project's augmentation is credited to. Findings: transfer learning and data
   augmentation **together** are what make small datasets work; neither alone
   suffices. With transfer learning, augmentation took accuracy 51.4 → 90.4 and
   F1 42.8 → 89.2. They freeze the VGG19 layers. Geometric augmentations are
   5–10 % of image size. Each vocalisation is augmented to 16 samples. They
   concatenate four auxiliary scalars (start/end time, min/max frequency) after
   the VGG19 output.

   **This project's v13 pipeline uses no augmentation at all.** Each clip becomes
   exactly one image. That places it in the quadrant the paper reports as
   insufficient.

2. **Sagar et al. (Gola, Sierra Leone/Liberia)** — BirdNET-based detectors for
   four primates including King Colobus. Methods worth copying: a random
   non-target class drawn from PAM recordings; seven iterative
   human-in-the-loop cycles sampling 200 detections ≥0.90 per round; a
   reliability curve fitted from 120 detections sampled uniformly across
   confidence; one target sound type per species chosen by explicit
   detectability criteria. They denoise focal recordings with spectral gating
   before overlaying PAM background. Their black-and-white colobus class was
   **3 recordings, 28 annotated vocalisations**, and it worked.

   Denoising was attempted here twice and rejected on measurement both times:
   the roar band's share of the clip **fell** in both attempts, meaning the
   gating was removing the call along with the background.

## The question

Given all of the above, what should be done next, and in what order?

Specific decisions that are open:

1. The 17 101 unlistened BirdNET clips are 65 % of the negative class. Keep,
   sample-and-check, or remove?
2. Should augmentation be added to the v13 pack path, given Sun et al. report it
   as necessary and this pipeline omits it entirely?
3. `balanced` class weighting suits re-ranking and appears to suit scanning
   badly. Is there a weighting that serves both, or do these need to be two
   models?
4. Is the re-ranker framing the right one for the paper, given the method cannot
   scan raw audio?
5. The Colobus class has no field-verified positive at these stations. Nine field
   clips from another project exist and are used as a test set. Should they enter
   training under leave-one-out instead?

Constraints: CPU only, 7–13 hours per sweep, one sweep at a time, and the paper
cannot be revised after submission.
