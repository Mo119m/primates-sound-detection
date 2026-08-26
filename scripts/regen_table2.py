"""Rewrite Table 2's body from the run that produced it.

The table was hand-transcribed once and then the dataset changed underneath it.
Eighteen rows of six numbers is not something to retype: this reads the run CSV
and emits the rows, so the table and the run cannot drift apart again.

It replaces only the lines between the first \\midrule after \\toprule and the
\\bottomrule, and it refuses to run if it cannot find exactly one such block, so
a moved table is a loud failure rather than a silent mis-edit.

The column order is fixed by the header and checked against it:

    Station | Detections | Deployed prec. | Threshold | Retrained prec.
            | FP removed (%) | Calls kept (%)

all of them the gated variants, because the deployed configuration is time-gated
to 05:00-19:00 and the ungated columns describe a pipeline nobody ran. Rows are
ordered by deployed precision ascending, which is how the table was built and
which puts the stations the deployed model handled worst at the top.
"""
import argparse
import os
import re
import sys

import pandas as pd

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DEFAULT_TEX = os.path.join(REPO, "overleaf", "methodsx_manuscript.tex")
DEFAULT_RUN = os.path.join(
    # The _evalfix rerun, not the original: the 2026-08-19 sweep scored the
    # augmented copies of 23 reviewed clips as detections (6,478 rows where
    # the review holds 6,110 originals) and every gated number moved with it.
    REPO, "data/outputs/v13_runs/full_2026-08-19/loso16_freqpos_evalfix.csv")

HEADER = ("Station & Detections & Deployed prec. & Threshold & Retrained prec. "
          "& FP removed (\\%) & Calls kept (\\%) \\\\")


def rows_from(run):
    d = pd.read_csv(run).sort_values("gated_v12_precision").reset_index(drop=True)
    out = []
    for _, r in d.iterrows():
        out.append(
            "{} & {:.0f} & {:.3f} & {:.3f} & {:.3f} & {:.1f} & {:.1f} \\\\".format(
                r.station, r.gated_detections, r.gated_v12_precision,
                r.gated_loso_threshold, r.gated_loso_precision,
                100 * r.gated_loso_fps_removed, 100 * r.gated_loso_calls_retained))
    macro = ("\\textbf{{Macro-average}} & --- & \\textbf{{{:.3f}}} & "
             "{:.3f}$^\\dagger$ & \\textbf{{{:.3f}}} & \\textbf{{{:.1f}}} & "
             "\\textbf{{{:.1f}}} \\\\").format(
                 d.gated_v12_precision.mean(),
                 d.gated_loso_threshold.median(),
                 d.gated_loso_precision.mean(),
                 100 * d.gated_loso_fps_removed.mean(),
                 100 * d.gated_loso_calls_retained.mean())
    return out, macro, d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tex", default=DEFAULT_TEX)
    ap.add_argument("--run", default=DEFAULT_RUN)
    ap.add_argument("--write", action="store_true",
                    help="apply the edit; without it, print the block and stop")
    a = ap.parse_args()

    body, macro, d = rows_from(a.run)
    block = "\n".join(body) + "\n\\midrule\n" + macro + "\n"

    text = open(a.tex, encoding="utf-8").read()
    # Anchor on the header line so this can only ever hit the LOSO table.
    pat = re.compile(
        r"(" + re.escape(HEADER) + r"\n\\midrule\n)(.*?)(\\bottomrule)",
        re.DOTALL)
    hits = pat.findall(text)
    if len(hits) != 1:
        sys.exit("expected exactly one LOSO table body, found {}".format(len(hits)))

    old_body = hits[0][1]
    if old_body == block:
        print("table already matches the run; nothing to do")
        return

    print("--- replacing {} lines with {} ---".format(
        len(old_body.strip().split("\n")), len(block.strip().split("\n"))))
    print(block)
    print("run provenance to state in the caption:")
    print("  file            {}".format(os.path.relpath(a.run, REPO).replace(os.sep, "/")))
    print("  folds           {}".format(len(d)))
    # The call count has to be the gated one to sit beside gated detections.
    # `calls` is the ungated total and differs where the time gate bites: at
    # IPA4ST it is 101 against 92 inside the window. Mixing the two would put a
    # base rate in the caption that is wrong in the flattering direction.
    gated_calls = (d.gated_v12_precision * d.gated_detections).sum()
    print("  review set      {:.0f} gated detections, {:.0f} of them calls "
          "(ungated: {:.0f} and {:.0f})".format(
              d.gated_detections.sum(), gated_calls,
              d.detections.sum(), d.calls.sum()))
    print("  threshold range {:.3f} to {:.3f}, {} of {} below 0.9, {} below 0.5".format(
        d.gated_loso_threshold.min(), d.gated_loso_threshold.max(),
        int((d.gated_loso_threshold < 0.9).sum()), len(d),
        int((d.gated_loso_threshold < 0.5).sum())))

    if not a.write:
        print("\n(dry run; pass --write to apply)")
        return

    new = pat.sub(lambda m: m.group(1) + block + m.group(3), text, count=1)
    open(a.tex, "w", encoding="utf-8", newline="").write(new)
    print("\nwritten to {}".format(a.tex))


if __name__ == "__main__":
    main()
