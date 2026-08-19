"""Audit the manuscript's labels and cross-references.

Written after a new table was given a label that already existed. LaTeX does not
stop for that -- it warns about a multiply-defined label and then resolves every
reference to whichever definition came last, so the text goes on compiling and
points at the wrong table. The structural checker did not catch it either,
because the environments were balanced and the columns matched; only the names
collided.

Three questions, all of which have a wrong answer that compiles:

    duplicate labels     a reference silently resolves to the last one
    references with no label   renders as ?? and is easy to miss in a long PDF
    labels never referenced    usually harmless, occasionally a deleted section

The third is reported and not counted as a failure.
"""
import collections
import os
import re
import sys

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DEFAULT = os.path.join(REPO, "overleaf", "methodsx_manuscript.tex")

LABEL = re.compile(r"\\label\{([^}]+)\}")
REF = re.compile(r"\\(?:page)?ref\{([^}]+)\}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    s = open(path, encoding="utf-8").read()

    labels = LABEL.findall(s)
    refs = set(REF.findall(s))
    dup = {k: v for k, v in collections.Counter(labels).items() if v > 1}
    dangling = sorted(refs - set(labels))
    unused = sorted(set(labels) - refs)

    print(f"{len(labels)} labels, {len(refs)} referenced")

    bad = 0
    if dup:
        bad += 1
        print(f"\nFAIL duplicate labels ({len(dup)}):")
        for k, v in sorted(dup.items()):
            print(f"  {k} defined {v} times")
            for m in LABEL.finditer(s):
                if m.group(1) == k:
                    print(f"    line {s[:m.start()].count(chr(10)) + 1}")
    if dangling:
        bad += 1
        print(f"\nFAIL references with no label ({len(dangling)}):")
        for k in dangling:
            print(f"  {k}")
    if unused:
        print(f"\nnote: {len(unused)} labels never referenced: "
              f"{', '.join(unused[:8])}{' ...' if len(unused) > 8 else ''}")

    print("\nPASS" if not bad else "\nFAIL")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
