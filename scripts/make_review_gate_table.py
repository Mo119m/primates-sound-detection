"""Build the smallest review table the time gate needs, with no coordinates in it.

`cleanup_vs_review.csv` is what makes the `gated_*` columns computable, and it
cannot leave this machine: 2,882 of its `file` cells and 2,882 of its
`recording` cells carry the recorder's position to about eleven metres, for a
primate that is hunted in Gabon. Without it a run prints "gated columns
unavailable" and writes a CSV missing the four columns every comparison in the
paper is quoted from -- which is exactly what three Colab folds did on
2026-08-24 before anyone noticed they were unusable.

So this writes the subset that the gate actually reads. `detection_hours()`
uses three columns and no others: `file` to join on, and `timestamp` + `start_s`
to place the detection in the day. Everything else -- the verdicts, the filter
flags, the Mahalanobis distances, the recording identifier -- stays here.

The join key is canonicalised, which is safe rather than merely convenient:
stripping the coordinate substring from all 6,189 filenames produces 6,189
distinct names, no collisions, and matches the same 6,478 index rows the raw
names match. That was measured, not assumed; a canonical key that silently
merged two recorders would corrupt every gated number downstream rather than
fail loudly.
"""
import argparse
import os
import re

import pandas as pd

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FULL = os.path.join(REPO, "data/outputs/auto_cleanup/cleanup_vs_review.csv")
OUT = os.path.join(REPO, "data/outputs/auto_cleanup/review_gate_table.csv")

GPS = re.compile(r"[+-]\d{2}\.\d{4}[+-]\d{3}\.\d{4}")
# Two patterns on purpose. GPS above is the STRIPPER, and its widths are
# load-bearing: canonicalising all 6,189 filenames with it produces 6,189
# distinct names with no collisions, which is what makes the join safe, so
# it is not widened. ANY_COORD below is the GUARD, and a guard should be as
# wide as it can be. They were the same pattern until 2026-08-31, when an
# audit found the refusal check below passing +000.0000+000.0000 straight
# through -- three digits of latitude, so the stripper's \d{2} misses it.
# Nothing sensitive escaped, that token is the null island a recorder
# writes when it never gets a fix, but the check said "refuse to write a
# coordinate, whatever column it turns up in" and did not.
ANY_COORD = re.compile(r"[+-]\d+\.\d+[+-]\d+\.\d+")
KEEP = ["file", "timestamp", "start_s"]


def canon(name):
    """The filename with the recorder's coordinates removed."""
    return GPS.sub("", str(name))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", default=FULL)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    rev = pd.read_csv(a.full)
    missing = [c for c in KEEP if c not in rev.columns]
    if missing:
        raise SystemExit("review table is missing {}".format(missing))

    small = rev[KEEP].copy()
    small["file"] = small["file"].map(canon)

    # Refuse to write a key that lost information. Downstream this table is
    # joined one-to-one; a duplicated key would quietly take whichever row
    # pandas happened to keep and put a wrong hour on a real detection.
    if small["file"].duplicated().any():
        dupes = small.loc[small["file"].duplicated(keep=False), "file"].head()
        raise SystemExit(
            "canonicalising the filenames collided: {} duplicate keys, e.g.\n{}"
            .format(int(small["file"].duplicated().sum()), dupes.to_string()))

    # And refuse to write anything coordinate-shaped, whatever column it turns
    # up in and whatever its digit widths are -- with one exemption, by exact
    # value and never by width: +000.0000+000.0000 is what a recorder writes
    # when it never got a fix. It is the absence of a position rather than one,
    # it is in 50 files that appear at two stations, and stripping it instead
    # would change the join keys that were verified to give 6,189 distinct
    # names. Exempting a value cannot let a real coordinate through; widening a
    # pattern can, which is how this check passed that token until 2026-08-31.
    NO_FIX = "+000.0000+000.0000"

    def _real_coord(x):
        return bool(ANY_COORD.search(str(x).replace(NO_FIX, "")))

    for c in small.columns:
        hits = small[c].astype(str).map(_real_coord).sum()
        if hits:
            raise SystemExit(
                "column {!r} still carries {} coordinates".format(c, int(hits)))

    small.to_csv(a.out, index=False)
    print("wrote {}".format(os.path.relpath(a.out, REPO).replace(os.sep, "/")))
    print("  {} rows, {} columns, {:.1f} KB".format(
        len(small), len(small.columns), os.path.getsize(a.out) / 1024))
    print("  {} unique join keys, no collisions".format(small["file"].nunique()))
    print("  dropped {} columns, including every verdict and both columns that "
          "carried coordinates".format(len(rev.columns) - len(KEEP)))
    print()
    print("This file is safe to put on Drive. cleanup_vs_review.csv is not.")


if __name__ == "__main__":
    main()
