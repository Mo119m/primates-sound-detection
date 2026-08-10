# V13 immutable experiment runbook

## Purpose and scope

This runbook governs all post-audit V13 experiments. A result is reportable
only when its exact data, feature artifacts, configuration, code revision, and
outputs are tied together by an integrity receipt.

The primary V13 task is **candidate re-ranking / review prioritisation**: given
the daytime candidates emitted by V12, rank or reject them before human review.
Its validation population is the reviewed V12 candidate set. It does not
estimate recall in continuous audio and must not be presented as a raw-audio
scanner result.

Use one immutable manifest per run. Start from
[`templates/v13_experiment_manifest.template.json`](templates/v13_experiment_manifest.template.json),
fill every `REQUIRED` value, then save the completed file in the run directory.
Never edit a completed manifest; make a new run ID and manifest instead.

## Run layout

Each run owns all of its derived outputs. Do not write results, checkpoints, or
receipts into shared `data/outputs/v13_loso*` or `data/outputs/v13_heads` paths.

```text
data/outputs/v13_runs/<run_id>/
  experiment_manifest.json       # immutable, completed template
  integrity_receipt.json         # written by the artifact gate
  command.txt                    # exact invoked command(s)
  environment.txt                # Python/package versions
  results.csv                    # fold-level metrics
  summary.json                   # aggregate metrics and definitions
  heads/                         # only this run's fold weights
  logs/                          # stdout/stderr
```

`<run_id>` uses `YYYYMMDDThhmmssZ_<task>_<short-description>`; for example,
`20260808T180000Z_reranker_verified_primary`.

## The artifact-hash gate (mandatory)

Before training, run the integrity checker against the frozen input artifacts.
It must write `integrity_receipt.json` in the run directory and exit non-zero
if any required check fails. The trainer must receive the frozen input paths and
its output metadata must repeat their hashes. Do not run after a failed check.

The gate must verify:

1. SHA-256, byte size, row count, and canonical row-order fingerprints for the
   manifest, index, image pack, and feature cache;
2. the complete per-row identity (`path`, `label`, `station`,
   `possible_stations`, `source`, `verified`, `period`, and `aug`), row address,
   uniqueness, and contiguous usable image/cache rows;
3. image/cache shape and dtype, plus an absence of unfinished `.partial` cache
   files; and
4. the label, verified-status, and source counts needed to make filtering
   auditable.

The checker cannot infer the intended model settings. The **release gate** also
requires an exact match between the frozen experiment manifest and the trainer
metadata for preprocessing, VGG tap, class order, pooling, detection groups,
time window, threshold rule, split rule, random seeds, code revision, and
trainer/model/config hashes. A mismatch invalidates the run.

The receipt is evidence, not a cache. A changed artifact requires a fresh run
ID, receipt, and training run. Do not override a failed check, trim files to
make row counts agree, or reuse heads/results from another receipt.

### Cache preparation is not a training run

If the checker reports a stale or missing feature cache after a newly packed
manifest/index/image set, build only that cache first. Use a fresh cache path
inside the eventual run directory and the cache-only flag below. It validates
the CSV/image contract, builds or validates the cache, records its inputs, then
exits before a fold, result CSV, or head is written. It deliberately refuses to
delete or reuse a `.partial` cache or a cache-build lock: those paths may belong
to another process. Choose a fresh run ID/path instead.

```powershell
$PY = 'C:/Users/Fudap/miniconda3/envs/primates/python.exe'
$RUN = 'data/outputs/v13_runs/20260808T180000Z_reranker_verified_primary'
if (Test-Path $RUN) { throw "Run directory already exists: $RUN" }
New-Item -ItemType Directory $RUN, "$RUN/logs" | Out-Null
& $PY scripts/train_v13_loso.py `
  --prepare-cache-only `
  --manifest 'data/outputs/v13_manifest.csv' `
  --index 'data/outputs/v13_index.csv' `
  --images 'data/outputs/v13_images.npy' `
  --cache "$RUN/v13_features.npy" `
  --out "$RUN/results.csv" `
  --run-metadata "$RUN/cache-prep.run.json"
```

Run the mandatory artifact gate only after cache preparation exits successfully.
The cache-preparation metadata is not an integrity receipt and cannot be used to
start a verified-only training run by itself.

### Commands once the guard is available

Continue from the fresh `$RUN` created in the cache-preparation command above.
Run these from the repository root after copying and completing the template.
First use the checker report to fill the template's artifact hash/count fields;
then freeze the copied manifest and rerun the checker to produce the final
receipt. Replace the data paths only with the paths registered in that manifest.

```powershell
$PY = 'C:/Users/Fudap/miniconda3/envs/primates/python.exe'
Copy-Item 'docs/templates/v13_experiment_manifest.template.json' "$RUN/experiment_manifest.json"
# Edit only the copied manifest, then freeze it.

# First prepare the new run-specific cache (see the cache-recovery command above).

& $PY scripts/check_v13_artifacts.py `
  --manifest 'data/outputs/v13_manifest.csv' `
  --index 'data/outputs/v13_index.csv' `
  --images 'data/outputs/v13_images.npy' `
  --cache "$RUN/v13_features.npy" `
  --full-hash `
  --json-out "$RUN/integrity_receipt.json" `
  --write-lock "$RUN/artifact.lock.json" `
  2>&1 | Tee-Object "$RUN/logs/integrity-check.log"

& $PY scripts/train_v13_loso.py `
  --verified-only `
  --manifest 'data/outputs/v13_manifest.csv' `
  --images 'data/outputs/v13_images.npy' `
  --index 'data/outputs/v13_index.csv' `
  --cache "$RUN/v13_features.npy" `
  --artifact-lock "$RUN/artifact.lock.json" `
  --folds all --epochs 15 --seed 42 --pooling temporal_freqpos `
  --out "$RUN/results.csv" --head-dir "$RUN/heads" `
  --run-metadata "$RUN/train.run.json" `
  2>&1 | Tee-Object "$RUN/logs/train.log"
```

Record the two expanded commands, `git rev-parse HEAD`, and the interpreter
plus package versions. Compare `train.run.json` with the frozen manifest before
reporting. If either command is not yet implemented, do not substitute an
unchecked legacy run for the primary experiment.

## Primary experiment: verified-only candidate re-ranker

This is the single prespecified main result.

- Task: `candidate_reranker` only.
- Evaluation: station-held-out, daytime reviewed V12 candidates only.
- Eligibility: labels explicitly marked human-verified in the frozen data
  manifest. Unlistened BirdNET detections and model-screened random-background
  windows are excluded.
- Split: leave one station out. A row is withheld when the held-out station is
  present in `possible_stations`; no ambiguous row may enter that fold's
  training set.
- Threshold: fit only on eligible non-held-out calibration data, never on the
  held-out station. Use the registered grouped decision rule and time window.
- Head/model: keep the registered trunk, tap layer, pooling, class order, loss,
  and seeds fixed. Any change is a separate experiment, not a rerun.
- Reporting: say “candidate precision”, “calls retained among V12 candidates”,
  and “false positives removed among V12 candidates”. Do not call calls
  retained “field recall” or claim continuous-audio detection performance.

No random-background inclusion, BirdNET inclusion, class-weight change,
time-window change, or threshold-policy change belongs in this primary result
unless it was registered before the run. The frozen manifest currently carries
an ``aug`` field: its exact distribution must be recorded in the receipt and
run metadata. A no-augmentation comparison is a separate registered artifact
set (or an explicit, audited ``aug == 0`` selection), not something to infer
from the name ``--verified-only``. Likewise, ``--verified-only`` does not imply
a four-class head: the exact selected class order (including any
``C_pogonias`` policy) must be frozen from the trainer metadata before results
are interpreted.

## Prespecified weak-negative ablation

Run only after the verified-only primary result has a passing receipt and a
completed 16-station sweep. Create a new run ID with
`experiment_kind: weak_negative_ablation`.

The ablation differs from the primary run in **one declared field only**:
`weak_negative_policy`. It may include only an audited, frozen list of weak
negatives whose audit protocol, verdict categories, sample frame, sample size,
and source hashes are recorded in the manifest. It must:

- retain every verified-only primary row and all primary hyperparameters;
- keep weak negatives marked `weak`, never relabel them `verified`;
- preserve the same LOSO exclusion rule and never mix a held-out station's weak
  negatives into that fold's training partition;
- report weak-negative count and provenance by source and class in `summary.json`; and
- compare only with the matching primary run ID and receipt.

It is a sensitivity analysis, not replacement training data and not a basis for
retrospectively changing the primary result.

## Scanner experiments are a separate study

A raw-audio scanner has a different data-generating population and success
criterion. It needs independently, exhaustively annotated continuous
recordings, stratified by station and time, with event-level recall, precision,
and false positives per hour. Candidate-only labels cannot validate it.

Use a distinct `task: continuous_audio_scanner`, artifact set, output root, and
manifest/receipt. Do not assemble or run candidate-reranker heads as a scanner
and do not merge scanner metrics into the candidate-reranker table. Scanner
experimentation starts only after its own annotation protocol and evaluation
manifest pass their hash gate.

## Before calling a run complete

- [ ] Completed manifest has no `REQUIRED` or placeholder values.
- [ ] Artifact gate passed and receipt is in the run directory.
- [ ] Run directory contains commands, logs, environment, results, summary,
  and fold heads.
- [ ] Every metric is labelled as candidate-set performance, not field recall.
- [ ] The run did not overwrite or reuse artifacts from another run.
- [ ] The result can be regenerated from the manifest and receipt alone.
