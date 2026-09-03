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
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
