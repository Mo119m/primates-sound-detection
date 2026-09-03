"""Count the manuscript sections MethodsX caps, and fail when one is over.

The journal's template states the limits inside this very file -- 200 words for
the abstract, 500 for Background -- and on 2026-08-31 a submission audit found
the abstract at 422 and Background at 615. Both had been over for weeks. A cap
written in a comment is not a check, and these are the two limits a technical
screen enforces before an editor reads a word.

Counting LaTeX is not counting text, so the rules here are explicit rather than
clever: strip %% comments, drop \\begin/\\end markers, drop control sequences and
their bracketed options, then count whitespace-separated tokens that contain at
least one alphanumeric character. A bracketed placeholder like
[Co-author Name] counts, because it will be words by the time this is submitted.

    python scripts/check_methodsx_limits.py
"""
import os
import re
import sys

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TEX = os.path.join(REPO, "overleaf/methodsx_manuscript.tex")

# name -> (start marker, end marker, cap). The caps are the journal's, quoted
# in the template comments at the head of the manuscript.
SECTIONS = [
    ("abstract", r"\begin{abstract}", r"\end{abstract}", 200),
    ("background", r"\section*{Background}", r"\section*{Method details}", 500),
]


def word_count(text):
    text = re.sub(r"%%.*", "", text)
    text = re.sub(r"\\(begin|end)\{[^}]*\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", text)
    text = re.sub(r"[{}$&~\\]", " ", text)
    return sum(1 for w in text.split() if any(c.isalnum() for c in w))


# Phrases that narrate the manuscript's own revision history. A reader can check
# a claim about the evidence and cannot check a claim about a draft they never
# saw, so these read as a research diary rather than a method -- and a
# submission audit on 2026-08-31 named them the clearest desk-reject lever in
# the file. Seven were removed that day; several had been added in the four
# days before it.
#
# The line is narrow. Reporting a SUPERSEDED RESULT is fine and stays: the
# withdrawn 98.12 % accuracy, a figure that came from an evaluation pool a later
# fix retired. So is self-criticism of the ANALYSIS -- "we now judge the
# criteria too weak". What is refused is the reference to a previous draft.
DRAFTING_HISTORY = [
    "earlier version of this",
    "earlier draft",
    "we first wrote",
    "than we wrote it",
    "first write-up",
    "this section overstated",
    "used to read",
]


def _drafting_history(tex):
    """Body only. A %% comment cannot desk-reject anything -- but it can be
    pasted into text that can, so comments are reported separately rather than
    ignored."""
    body, comments = [], []
    for i, line in enumerate(tex.splitlines(), 1):
        (comments if line.lstrip().startswith("%") else body).append((i, line))
    hits = {"body": [], "comment": []}
    for where, lines in (("body", body), ("comment", comments)):
        for i, line in lines:
            low = line.lower()
            for p in DRAFTING_HISTORY:
                if p in low:
                    hits[where].append((i, p, line.strip()[:90]))
    return hits


def main():
    tex = open(TEX, encoding="utf-8").read()
    bad = 0
    for name, start, end, cap in SECTIONS:
        a = tex.find(start)
        b = tex.find(end, a + 1) if a >= 0 else -1
        if a < 0 or b < 0:
            print(f"  {name:12s} NOT FOUND ({start!r} .. {end!r})")
            bad += 1
            continue
        n = word_count(tex[a:b])
        over = n - cap
        flag = "OK " if over <= 0 else "OVER"
        print(f"  {flag} {name:12s} {n:4d} words   cap {cap}"
              + (f"   ({over:+d})" if over > 0 else ""))
        bad += over > 0

    # The submission class, which is 'review' for Elsevier at submission and
    # 'final,3p' only for the accepted version. Getting this wrong is a
    # formatting rejection, not a scientific one, and it is one word.
    m = re.search(r"\\documentclass\[([^\]]*)\]", tex)
    cls = m.group(1) if m else ""
    ok = "review" in cls
    print(f"  {'OK ' if ok else 'OVER'} documentclass  [{cls}]"
          + ("" if ok else "   should be [review,...] at submission"))
    bad += not ok

    hits = _drafting_history(tex)
    if hits["body"]:
        bad += 1
        print(f"  OVER drafting history  {len(hits['body'])} sentence(s) in the body")
        for i, p, line in hits["body"]:
            print(f"       line {i}: {p!r} -- {line}")
        print("       A reader cannot check a claim about a draft. Report the"
              " superseded RESULT instead, which they can.")
    else:
        print("  OK  drafting history  none in the body")
    if hits["comment"]:
        print(f"  --  ({len(hits['comment'])} in %% comments, not counted:"
              f" they do not render, but they do get pasted)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
