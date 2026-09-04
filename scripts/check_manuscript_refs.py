"""Fail when a citation, label or figure the manuscript names does not exist.

Written on 2026-08-31, after the abstract and Background were each cut by
roughly a third in one sitting. That kind of surgery is where a \\cite whose
bibliography entry lived in the deleted paragraph survives, or a
Table~\\ref{...} points at a label that moved. Neither shows up in a word count
and neither is caught by the numeric verifier, which only ever reads the .tex
to confirm a string is present.

Three things are checked, all of them mechanical:

  citations   every \\cite key resolves to a \\bibitem or a .bib entry, and every
              defined entry is actually cited. An uncited entry is not an error
              in itself, but after a large cut it is usually the trace of a
              paragraph that went.
  labels      every \\ref and \\autoref resolves to a \\label, and every label is
              referenced. Elsevier renders an unresolved reference as "??".
  graphics    every \\includegraphics file exists on disk.

    python scripts/check_manuscript_refs.py
"""
import os
import re
import sys

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TEX = os.path.join(REPO, "overleaf/methodsx_manuscript.tex")
OVERLEAF = os.path.join(REPO, "overleaf")


def strip_comments(text):
    # A % that is not escaped starts a comment. Citations inside commented-out
    # paragraphs must not count as used -- that is exactly how a stale \cite
    # hides.
    out = []
    for line in text.splitlines():
        m = re.search(r"(?<!\\)%", line)
        out.append(line[: m.start()] if m else line)
    return "\n".join(out)


def main():
    raw = open(TEX, encoding="utf-8").read()
    tex = strip_comments(raw)
    bad = 0

    # ---- citations ----
    cited = set()
    for m in re.finditer(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]*)\}", tex):
        cited.update(k.strip() for k in m.group(1).split(",") if k.strip())

    defined = set(re.findall(r"\\bibitem(?:\[[^\]]*\])?\{([^}]*)\}", tex))
    bibs = [f for f in os.listdir(OVERLEAF) if f.endswith(".bib")]
    for b in bibs:
        body = open(os.path.join(OVERLEAF, b), encoding="utf-8",
                    errors="ignore").read()
        defined.update(m.strip() for m in
                       re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", body))

    missing = sorted(cited - defined)
    unused = sorted(defined - cited)
    print(f"  citations: {len(cited)} cited, {len(defined)} defined"
          + (f", {len(bibs)} .bib file(s)" if bibs else ", inline bibliography"))
    if missing:
        bad += 1
        print(f"  OVER cited but never defined: {missing}")
    if unused:
        print(f"  --   defined but never cited: {unused}"
              "   (often the trace of a deleted paragraph)")

    # ---- labels ----
    labels = set(re.findall(r"\\label\{([^}]*)\}", tex))
    refs = set()
    for m in re.finditer(r"\\(?:auto|c|C)?ref\*?\{([^}]*)\}", tex):
        refs.update(k.strip() for k in m.group(1).split(",") if k.strip())
    dangling = sorted(refs - labels)
    orphan = sorted(labels - refs)
    print(f"  labels: {len(labels)} defined, {len(refs)} referenced")
    if dangling:
        bad += 1
        print(f"  OVER referenced but never defined: {dangling}"
              "   (Elsevier renders these as ??)")
    if orphan:
        print(f"  --   defined but never referenced: {orphan}")

    # ---- graphics ----
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", tex)
    absent = []
    for f in figs:
        cands = [os.path.join(OVERLEAF, f)]
        if "." not in os.path.basename(f):
            cands += [os.path.join(OVERLEAF, f + e)
                      for e in (".pdf", ".png", ".jpg", ".eps")]
        if not any(os.path.exists(c) for c in cands):
            absent.append(f)
    print(f"  graphics: {len(figs)} included")
    if absent:
        bad += 1
        print(f"  OVER included but not on disk: {absent}")

    if not bad:
        print("  every citation, reference and figure resolves")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
