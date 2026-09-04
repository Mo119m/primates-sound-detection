#!/usr/bin/env bash
# Train one frozen-trunk arm over the sixteen LOSO folds.
#
# Every argument except --trunk, --cache, --out and --head-dir is copied
# verbatim from loso16_freqpos_evalfix.run.json, the run the paper reports, so
# the backbone is the only thing that differs between this and the incumbent.
#
#   scripts/run_trunk_arm.sh <trunk> <draw>
#
# The cache must already exist:
#   python scripts/train_v13_loso.py --prepare-cache-only --trunk <trunk> \
#       --manifest .../manifest.csv --index .../v13_index.csv \
#       --images .../v13_images.npy --cache .../feats_<trunk>.npy
set -euo pipefail

TRUNK="${1:?usage: run_trunk_arm.sh <trunk> <draw>}"
DRAW="${2:?usage: run_trunk_arm.sh <trunk> <draw>}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO/data/outputs/v13_runs/full_2026-08-19"
OUT="$REPO/data/outputs/v13_runs/trunks_2026-09-04"
PY="${PYTHON:-/c/Users/Fudap/miniconda3/envs/primates/python.exe}"

FOLDS="IPA1ST,IPA2ST,IPA4ST,IPA6ST,IPA7ST,IPA8ST,IPA10ST,IPA11ST,IPA13ST,IPA14ST,IPA15ST,IPA16ST,IPA17ST,IPA18ST,IPA19ST,IPA20ST"
CACHE="$OUT/feats_${TRUNK}.npy"

if [ ! -f "$CACHE" ]; then
  echo "missing feature cache: $CACHE" >&2
  exit 1
fi

mkdir -p "$OUT"
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONIOENCODING=utf-8

exec "$PY" "$REPO/scripts/train_v13_loso.py" \
  --folds "$FOLDS" \
  --epochs 15 \
  --patience 3 \
  --overwrite \
  --keep-all-background \
  --pooling temporal_freqpos \
  --trunk "$TRUNK" \
  --manifest "$SRC/manifest.csv" \
  --index "$SRC/v13_index.csv" \
  --images "$SRC/v13_images.npy" \
  --cache "$CACHE" \
  --out "$OUT/loso16_${TRUNK}_${DRAW}.csv" \
  --head-dir "$OUT/heads_${TRUNK}_${DRAW}" \
  --run-metadata "$OUT/loso16_${TRUNK}_${DRAW}.run.json"
