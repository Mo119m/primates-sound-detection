"""
Turn a folder of clips into a listening job with somewhere to put the answers.

Handing someone a folder of WAVs and asking them to listen produces an opinion.
Handing them a sheet with one row per clip produces a number, and a number is
what both of the open questions need: how often the detector fires on audio
nobody selected, and how often it is right when it does.

The sheet is a CSV with the verdict column blank. Open it in a spreadsheet, play
the clips in order, type in the verdict, save. Nothing else needs to be tracked:
the confidence, the station and the source recording are already in the row.

    python scripts/make_listening_sheet.py --folder data/outputs/random_mine_suspect --out random_sample.csv
"""
import argparse
import glob
import os
import re
import sys

import pandas as pd

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verdicts", default="call,noise,unsure",
                    help="Allowed answers, listed in the header comment.")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.folder, "**", "*.wav"),
                             recursive=True))
    if not files:
        sys.exit(f"no clips under {args.folder}")

    rows = []
    for p in files:
        name = os.path.basename(p)
        rel = os.path.relpath(p, REPO)
        parent = os.path.basename(os.path.dirname(p))
        conf = re.search(r"conf([0-9.]+)", name)
        st = re.search(r"(ipa\d+st)", parent, re.I) or re.search(r"(IPA\d+ST)", rel)
        rows.append({
            "clip": name,
            "path": rel,
            "folder": parent,
            "station": st.group(1).upper() if st else "",
            "model_confidence": float(conf.group(1)) if conf else "",
            "verdict": "",
            "species_if_call": "",
            "note": "",
        })
    d = pd.DataFrame(rows)

    # Shuffle within the sheet so the listener is not primed by confidence
    # order. A run of high-confidence clips first teaches the ear what to expect
    # and the later verdicts stop being independent.
    d = d.sample(frac=1.0, random_state=0).reset_index(drop=True)
    d.insert(0, "n", range(1, len(d) + 1))

    out = os.path.join(REPO, args.out) if not os.path.isabs(args.out) else args.out
    with open(out, "w", encoding="utf-8", newline="") as fh:
        fh.write(f"# verdict: one of {args.verdicts}\n")
        fh.write(f"# species_if_call: Cernic / C_pogonias / Colobus_guereza / "
                 f"other -- only if verdict is 'call'\n")
        fh.write(f"# rows are shuffled on purpose; do not sort by confidence "
                 f"before listening\n")
        d.to_csv(fh, index=False)
    print(f"{len(d)} clips -> {os.path.relpath(out, REPO)}")
    if "model_confidence" in d and d.model_confidence.astype(str).str.len().max():
        known = pd.to_numeric(d.model_confidence, errors="coerce").dropna()
        if len(known):
            print(f"  confidence range {known.min():.3f}-{known.max():.3f}")
    print(f"  stations: {d.station.nunique()}")


if __name__ == "__main__":
    main()
