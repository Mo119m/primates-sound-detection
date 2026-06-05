"""
Tune the detection operating point (confidence / Mahalanobis) on DEV stations.

The model fires on a 2s window when its softmax confidence exceeds
``DETECTION_CONFIDENCE_THRESHOLD`` (0.4). That threshold was never calibrated
against field labels, so it lets through a lot of low-confidence false alarms.
This script sweeps the threshold over a *labelled* subset of dev-station
detections and reports the precision / recall trade-off so a defensible
operating point can be chosen -- WITHOUT touching the held-out test stations
(IPA19/20 are hard-refused).

Where the labels come from
--------------------------
We have no exhaustive per-clip ground truth, but the field reviewer has
confirmed enough to label a usable subset:

  * Real Cernic (positives): the *clean* Cernic detections at stations where the
    reviewer confirmed the clean bucket is genuine putty-nose calls
    (--cernic-real-stations, e.g. IPA15/17/18).
  * FP Cernic (negatives): every Cernic detection at stations the reviewer found
    to contain no real Cernic (--cernic-fp-stations, e.g. IPA13/14/16).
  * FP Colobus (negatives): every Colobus detection anywhere -- no real Colobus
    was found at any dev station.

Stations not named in either list are treated as UNKNOWN and excluded from the
labelled set, so an uncertain old station never contaminates the calibration.

Honest caveats (printed in the report too):
  * The positive set is small and drawn from the *clean* bucket, so it already
    has low Mahalanobis distance by construction -- the Mahalanobis sweep is
    therefore optimistic. The confidence sweep is the more meaningful one.
  * A threshold can only remove FPs that are *separable* in confidence /
    distance. The hard FPs (high-confidence, in-distribution -- e.g. the clean
    Cernic FPs at IPA19) are not separable by any threshold; if the
    distributions overlap heavily, that is the signal to change features
    instead of the operating point.

Input
-----
Per-station auto_cleanup CSVs written by run_auto_cleanup:
    {cleanup_root}/{station}/clean_detections.csv      (n_flags == 0)
    {cleanup_root}/{station}/suspicious_detections.csv (n_flags  > 0)
Both carry: species, confidence, mahalanobis_d2, n_flags, yamnet_top, source_file.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEST_TOKENS = ("19", "20")
CERNIC = "Cernic"
COLOBUS = "Colobus_guereza"


def _is_test_station(station: str) -> bool:
    return any(tok in station for tok in TEST_TOKENS)


def load_station(cleanup_root: str, station: str) -> pd.DataFrame:
    """Concatenate a station's clean + suspicious cleanup CSVs (whichever exist)."""
    frames = []
    for name in ("clean_detections.csv", "suspicious_detections.csv"):
        path = os.path.join(cleanup_root, station, name)
        if os.path.exists(path):
            df = pd.read_csv(path)
            if len(df):
                df["station"] = station
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_labeled(cleanup_root, real_stations, fp_stations) -> pd.DataFrame:
    """Assemble the labelled detection table (label 1 = real, 0 = FP)."""
    real_stations = set(real_stations)
    fp_stations = set(fp_stations)
    rows = []
    for station in sorted(real_stations | fp_stations):
        df = load_station(cleanup_root, station)
        if df.empty:
            print(f"   {station}: no cleanup CSVs found (skipped)")
            continue
        n_flags = df.get("n_flags", pd.Series(np.zeros(len(df))))
        # Cernic positives: clean bucket at confirmed-real stations.
        if station in real_stations:
            m = (df["species"] == CERNIC) & (n_flags == 0)
            pos = df[m].copy()
            pos["label"] = 1
            rows.append(pos)
        # Cernic negatives: all Cernic at confirmed-FP stations.
        if station in fp_stations:
            m = df["species"] == CERNIC
            neg = df[m].copy()
            neg["label"] = 0
            rows.append(neg)
        # Colobus negatives: all Colobus anywhere in the named stations.
        m = df["species"] == COLOBUS
        col = df[m].copy()
        col["label"] = 0
        rows.append(col)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out.drop_duplicates(subset=["station", "source_file", "start_time", "species"])


def _dist(name, values):
    if len(values) == 0:
        print(f"   {name}: (none)")
        return
    q = np.percentile(values, [0, 10, 25, 50, 75, 90, 100])
    print(f"   {name} (n={len(values)}): "
          f"min={q[0]:.3f} p10={q[1]:.3f} p25={q[2]:.3f} med={q[3]:.3f} "
          f"p75={q[4]:.3f} p90={q[5]:.3f} max={q[6]:.3f}")


def sweep_confidence(df_species: pd.DataFrame, species: str):
    """Print a confidence-threshold sweep for one species' labelled detections."""
    pos = df_species[df_species["label"] == 1]["confidence"].to_numpy()
    neg = df_species[df_species["label"] == 0]["confidence"].to_numpy()
    print(f"\n=== {species} ===")
    print(f"   positives (real): {len(pos)}   negatives (FP): {len(neg)}")
    _dist("confidence real", pos)
    _dist("confidence FP  ", neg)

    if len(pos) == 0 or len(neg) == 0:
        print("   (need both real and FP examples to sweep -- skipped)")
        return None

    print(f"\n   {'thresh':>7} {'recall':>8} {'FP kept':>8} {'FP cut%':>8} {'precision':>10}")
    best = None
    for t in np.round(np.arange(0.40, 1.001, 0.05), 2):
        kept_pos = int((pos >= t).sum())
        kept_neg = int((neg >= t).sum())
        recall = kept_pos / len(pos)
        fp_cut = 1.0 - kept_neg / len(neg)
        denom = kept_pos + kept_neg
        precision = kept_pos / denom if denom else float("nan")
        print(f"   {t:7.2f} {recall:8.2%} {kept_neg:8d} {fp_cut:8.1%} {precision:10.2%}")
        # Recommend the highest threshold that still keeps >=90% of real calls.
        if recall >= 0.90:
            best = (t, recall, fp_cut, precision)
    if best:
        print(f"\n   -> suggested confidence threshold {best[0]:.2f} "
              f"(keeps {best[1]:.0%} of real calls, cuts {best[2]:.0%} of FPs, "
              f"precision {best[3]:.0%} on the labelled set)")
    else:
        print("\n   -> no threshold keeps >=90% of real calls: real and FP "
              "confidences overlap too much (threshold tuning is limited here)")
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cleanup-root", required=True)
    p.add_argument("--cernic-real-stations", nargs="*", default=[],
                   help="stations whose CLEAN Cernic detections are confirmed "
                        "real (source of positives)")
    p.add_argument("--cernic-fp-stations", nargs="*", default=[],
                   help="stations with no real Cernic (all Cernic = FP)")
    args = p.parse_args()

    bad = [s for s in (args.cernic_real_stations + args.cernic_fp_stations)
           if _is_test_station(s)]
    if bad:
        p.error(f"refusing to use held-out test station(s): {', '.join(bad)}")
    if not args.cernic_real_stations or not args.cernic_fp_stations:
        p.error("need at least one --cernic-real-stations and one "
                "--cernic-fp-stations to calibrate Cernic")

    print("Building labelled dev set...")
    print(f"   real-Cernic stations: {args.cernic_real_stations}")
    print(f"   FP-Cernic stations:   {args.cernic_fp_stations}")
    df = build_labeled(args.cleanup_root, args.cernic_real_stations,
                       args.cernic_fp_stations)
    if df.empty:
        p.error("no labelled detections assembled -- check the cleanup CSVs exist")

    sweep_confidence(df[df["species"] == CERNIC], "Cernic")
    # Colobus has no real positives anywhere, so only its FP confidence
    # distribution is informative (how high do Colobus false alarms score?).
    col = df[(df["species"] == COLOBUS) & (df["label"] == 0)]["confidence"].to_numpy()
    print("\n=== Colobus_guereza (FP only -- no real Colobus at any dev station) ===")
    _dist("confidence FP", col)
    print("\nReminder: tune on these dev numbers, then APPLY the chosen "
          "threshold to IPA19/20 and report -- never tune on the test stations.")


if __name__ == "__main__":
    main()
