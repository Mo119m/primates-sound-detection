#!/usr/bin/env bash
# Run the remaining frozen-trunk arms one after another, unattended.
#
# Sequential on purpose. One 16-fold run already saturates this CPU at ~13x
# parallelism across 149 threads; a second concurrent run would not finish two
# arms in the time one takes, it would finish neither and make the per-fold
# timings useless for planning.
#
# Two draws per trunk, because training here is unseeded: --seed fixes the fold
# split and the inner train/val grouping, not weight initialisation or batch
# order. Three draws of one specification already measured a threshold-free
# spread of up to 0.0026 in the paired mean with |t| reaching 1.89, so a single
# draw of a new trunk is a coin, not a measurement.
#
#   nohup bash scripts/run_trunk_queue.sh > .../queue.log 2>&1 &
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO/data/outputs/v13_runs/trunks_2026-09-04"
SRC="$REPO/data/outputs/v13_runs/full_2026-08-19"
PY="${PYTHON:-/c/Users/Fudap/miniconda3/envs/primates/python.exe}"
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONIOENCODING=utf-8

stamp() { date -u +"%H:%M:%SZ"; }

# Wait for whatever is already training to finish writing its 16th head.
wait_for() {   # wait_for <head-dir> <label>
  local d="$1" label="$2" n
  while true; do
    n=$(ls "$d" 2>/dev/null | wc -l)
    [ "$n" -ge 16 ] && { echo "[$(stamp)] $label complete (16 folds)"; return 0; }
    sleep 60
  done
}

build_cache() {  # build_cache <trunk>
  local t="$1"
  if [ -f "$OUT/feats_${t}.npy" ]; then
    echo "[$(stamp)] cache for $t already present"
    return 0
  fi
  echo "[$(stamp)] building cache for $t"
  "$PY" "$REPO/scripts/train_v13_loso.py" --prepare-cache-only --trunk "$t" \
    --manifest "$SRC/manifest.csv" --index "$SRC/v13_index.csv" \
    --images "$SRC/v13_images.npy" --cache "$OUT/feats_${t}.npy" \
    > "$OUT/cache_${t}.log" 2>&1
  echo "[$(stamp)] cache for $t done ($(du -h "$OUT/feats_${t}.npy" 2>/dev/null | cut -f1))"
}

arm() {  # arm <trunk> <draw>
  local t="$1" d="$2"
  if [ "$(ls "$OUT/heads_${t}_${d}" 2>/dev/null | wc -l)" -ge 16 ]; then
    echo "[$(stamp)] ${t}_${d} already complete, skipping"
    return 0
  fi
  echo "[$(stamp)] starting ${t}_${d}"
  bash "$REPO/scripts/run_trunk_arm.sh" "$t" "$d" \
    > "$OUT/train_${t}_${d}.log" 2>&1
  echo "[$(stamp)] finished ${t}_${d} (rc=$?)"
}

echo "[$(stamp)] queue start"

# 1. the run already in flight
wait_for "$OUT/heads_effnetv2s_d1" "effnetv2s_d1"

# 2. its second draw -- the arm is a distribution, not a point
arm effnetv2s d2

# 3. a second, architecturally different trunk
build_cache convnext_tiny
arm convnext_tiny d1
arm convnext_tiny d2

echo "[$(stamp)] queue done. Score with:"
echo "  python scripts/prauc_trunk_arms.py --arms effnetv2s_d1,effnetv2s_d2,convnext_tiny_d1,convnext_tiny_d2"
