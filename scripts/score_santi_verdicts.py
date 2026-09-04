"""Score the species expert's returned listening sheets.

Two of the three packages came back on 2026-09-04, as Raven MANUAL ID exports
rather than the VERDICT_SHEET.csv that shipped with them. The vocabulary he
used is {C nic, Noise}: two labels, not the yes/no/unsure/nictitans the README
asked for. That is the one interpretive question in this file and it is handled
explicitly rather than silently --

  cer block   "is this Cercopithecus nictitans"        C nic -> yes,  Noise -> no
  pog block   "is this C. pogonias; if nictitans, say"  C nic -> nictitans,
                                                        Noise -> neither
  col block   "is a Colobus guereza roar audible"       Noise -> no

-- and the absence of an "unsure" category is reported, not assumed away. He
used "C nic" 28 times in the cer block and never once in the pog block, so the
distinction the pog question turns on was available to him and he declined it;
that is what makes the pogonias zero interpretable.

The clip names are blinded: they carry the species and nothing else. Station and
confidence live in key files that stay local because their source_file column
carries recorder coordinates.

    python scripts/score_santi_verdicts.py
"""
import io
import math
import os
import re
import sys

import pandas as pd

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
# The returned sheets are archived in the repo (under gitignored data/) so this
# does not depend on a Desktop that gets tidied. The Desktop originals are the
# fallback, named as they arrived.
_ARCH = os.path.join(REPO, "data/outputs/santi_verdicts_2026-09-04")
DESK = "C:/Users/Fudap/OneDrive/Desktop"


def _first(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]


PREC_SHEET = _first(os.path.join(_ARCH, "precision_150_checked.csv"),
                    os.path.join(DESK, "precision 150 checked.csv"))
RESC_SHEET = _first(os.path.join(_ARCH, "rescan_196_checked.csv"),
                    os.path.join(DESK, "FOR_SANTI_2026-08-28_Checked.csv"))
PREC_KEY = os.path.join(REPO, "data/outputs/precision_resample/sampling_key.csv")
RESC_KEY = os.path.join(REPO,
                        "data/outputs/detection_review/rescan_package_key_2026-08-28.csv")

QUESTION = {
    "Cernic": "is this Cercopithecus nictitans",
    "C_pogonias": "is this C. pogonias (if nictitans, say so)",
    "Colobus_guereza": "is a Colobus guereza roar audible",
}


def wilson(k, n, z=1.959964):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def rule_of_three(n):
    """Upper 95 % bound on a rate when zero of n were positive."""
    return 3.0 / n if n else float("nan")


def read_sheet(path):
    raw = open(path, encoding="utf-8-sig", newline="").read()
    df = pd.read_csv(io.StringIO(raw))
    df.columns = [c.strip() for c in df.columns]
    idcol = next(c for c in df.columns if c.upper().startswith("MANUAL ID"))
    out = pd.DataFrame({
        "clip": df["IN FILE"].astype(str).str.strip(),
        "verdict": df[idcol].astype(str).str.strip(),
    })
    return out


def main():
    for p in (PREC_SHEET, RESC_SHEET, PREC_KEY, RESC_KEY):
        if not os.path.exists(p):
            print(f"MISSING {p}")
            return 1

    bad = 0

    # ---------------------------------------------------------- precision 150
    print("=" * 72)
    print("PRECISION SAMPLE  (s0000-s0149)")
    print("  equal-probability draw of 150 of the 768 detections at or above")
    print("  0.95, seed 20260811 -- unweighted proportion is unbiased")
    s = read_sheet(PREC_SHEET)
    key = pd.read_csv(PREC_KEY)
    m = s.merge(key[["clip", "station"]], on="clip", how="left")
    if m.station.isna().any():
        print(f"  ! {int(m.station.isna().sum())} clips not in the sampling key")
        bad += 1
    unmatched = set(key["clip"]) - set(s["clip"])
    if unmatched:
        print(f"  ! {len(unmatched)} sampled clips have no verdict: "
              f"{sorted(unmatched)[:6]}")
        bad += 1
    yes = int((m.verdict == "C nic").sum())
    n = len(m)
    lo, hi = wilson(yes, n)
    print(f"  verdicts: {dict(m.verdict.value_counts())}")
    print(f"  precision {yes}/{n} = {yes / n:.4f}   Wilson 95% [{lo:.4f}, {hi:.4f}]")
    print("  per station (descriptive only -- the design pools):")
    for st, g in m.groupby("station"):
        gy = int((g.verdict == "C nic").sum())
        print(f"    {st:8s} {gy:3d}/{len(g):3d} = {gy / len(g):.3f}")

    # ------------------------------------------------------------ rescan 196
    print()
    print("=" * 72)
    print("RESCAN PACKAGE  (cer/pog/col, seed 20260828)")
    r = read_sheet(RESC_SHEET)
    rkey = pd.read_csv(RESC_KEY)
    rm = r.merge(rkey[["clip", "species", "station", "confidence"]],
                 on="clip", how="left")
    if rm.species.isna().any():
        print(f"  ! {int(rm.species.isna().sum())} clips not in the key")
        bad += 1
    miss = set(rkey["clip"]) - set(r["clip"])
    if miss:
        print(f"  ! {len(miss)} packaged clips have no verdict: {sorted(miss)[:6]}")
        bad += 1
    vocab = sorted(set(rm.verdict))
    print(f"  label vocabulary used: {vocab}")
    if "unsure" not in [v.lower() for v in vocab]:
        print("    note: no 'unsure' category. Confirmed deliberate on enquiry")
        print("    (2026-09-04) -- the expert double-checked rather than leaving")
        print("    anything unresolved, so the proportions carry no unsure rows.")

    for sp in ("Cernic", "C_pogonias", "Colobus_guereza"):
        g = rm[rm.species == sp]
        if not len(g):
            continue
        pos = int((g.verdict == "C nic").sum())
        print()
        print(f"  {sp}  ({QUESTION[sp]})")
        print(f"    {pos}/{len(g)} answered 'C nic'")
        if pos == 0:
            print(f"    zero positives: 95 % upper bound on the rate is "
                  f"{rule_of_three(len(g)):.3f} (rule of three)")
        else:
            lo2, hi2 = wilson(pos, len(g))
            print(f"    rate {pos / len(g):.4f}  Wilson 95% [{lo2:.4f}, {hi2:.4f}]")
        print("    per station:")
        for st, gg in g.groupby("station"):
            gp = int((gg.verdict == "C nic").sum())
            print(f"      {st:8s} {gp:3d}/{len(gg):3d}"
                  + (f"   (mean conf {gg.confidence.mean():.3f})"))

    # ------------------------------------------------------- still outstanding
    print()
    print("=" * 72)
    print("STILL OUTSTANDING")
    print("  d000-d116 (117 clips, the dawn-discarded package) has not come back.")
    print("  Nothing above depends on it.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
