"""Turn the expert's 150 verdicts into the paper's precision figure.

The design makes the arithmetic trivial, which is the point of the design:
every clip was drawn with equal probability 150/768 from the detections at or
above 0.95 (seed 20260811), so the unweighted proportion is unbiased and a
Wilson interval is valid. No weighting, no strata, nothing to reconstruct.
The 92.7 % this replaces had neither property.

`unsure` rows are the honest complication. They are reported, and the estimate
is given three ways: excluding them, counting them all as calls, counting them
all as errors. The last two bracket every possible resolution; if the bracket
is tight the unsures do not matter, and if it is wide the paper must say so
rather than pick a side.
"""
import argparse
import math
import os
import sys

import pandas as pd

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SHEET = "C:/Users/Fudap/OneDrive/Desktop/FOR_SANTI_2026-08-25/VERDICT_SHEET.csv"
KEY = os.path.join(REPO, "data/outputs/precision_resample/sampling_key.csv")


def wilson(k, n, z=1.959964):
    """95 % Wilson interval for k successes in n trials."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sheet", default=SHEET)
    ap.add_argument("--key", default=KEY)
    a = ap.parse_args()

    s = pd.read_csv(a.sheet)
    v = s["verdict_yes_or_no"].astype(str).str.strip().str.lower()
    yes = int((v == "yes").sum())
    no = int((v == "no").sum())
    unsure = int((v == "unsure").sum())
    blank = int((~v.isin(("yes", "no", "unsure"))).sum())

    print(f"verdicts: {yes} yes, {no} no, {unsure} unsure, {blank} blank/other "
          f"of {len(s)}")
    if blank:
        blanks = s.loc[~v.isin(("yes", "no", "unsure")), "clip"].tolist()
        print(f"  ! blank rows are indistinguishable from rows nobody reached, "
              f"which is the failure the old sample died of. Chase these: "
              f"{blanks[:8]}{' ...' if len(blanks) > 8 else ''}")
        sys.exit(1)

    for label, k, n in (
            ("excluding unsure", yes, yes + no),
            ("unsure counted as calls", yes + unsure, yes + no + unsure),
            ("unsure counted as errors", yes, yes + no + unsure)):
        lo, hi = wilson(k, n)
        print(f"  {label:26s} {k}/{n} = {k / n:.3f}  Wilson 95% [{lo:.3f}, {hi:.3f}]")

    # Per-station split, descriptive only: the sample was pooled by design and
    # 150 does not power a between-station comparison.
    if os.path.exists(a.key):
        key = pd.read_csv(a.key)
        m = s.merge(key[["clip", "station"]], on="clip", how="left")
        m["v"] = v.values
        print("\nper station, descriptive only (the design pools):")
        for st, g in m.groupby("station"):
            gy = int((g.v == "yes").sum())
            gn = int((g.v == "no").sum())
            print(f"  {st}: {gy}/{gy + gn} yes"
                  + (f" (+{int((g.v == 'unsure').sum())} unsure)"
                     if (g.v == "unsure").any() else ""))

    print("\nSentence for the manuscript (excluding-unsure form):")
    lo, hi = wilson(yes, yes + no)
    print(f"  Of an equal-probability sample of 150 of the 768 detections at or "
          f"above 0.95 (seed 20260811), {yes} of {yes + no} adjudicated clips "
          f"were genuine calls: precision {yes / (yes + no):.3f} "
          f"(95 % Wilson interval {lo:.3f}--{hi:.3f}).")


if __name__ == "__main__":
    main()
