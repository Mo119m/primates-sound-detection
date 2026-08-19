"""Structural sanity check on the manuscript, since there is no local LaTeX.

Not a substitute for compiling -- it catches the two mistakes that editing a
table by hand actually produces: an environment left unclosed, and a row whose
cell count disagrees with the column specification. Overleaf reports both as
errors far from where they were made.
"""
import os
import re
import sys

# Relative to this file. The scratch version hard-coded one developer's
# drive, which is fine for a throwaway and wrong once the paper depends
# on the check passing.
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "overleaf", "methodsx_manuscript.tex")
if len(sys.argv) > 1:
    PATH = sys.argv[1]
src = open(PATH, encoding="utf-8").read()
lines = src.split("\n")
bad = 0

for env in ["tabular", "tabularx", "center", "table", "figure", "document",
            "itemize", "enumerate", "abstract"]:
    b = len(re.findall(r"\\begin\{" + env + r"\}", src))
    e = len(re.findall(r"\\end\{" + env + r"\}", src))
    flag = "OK" if b == e else "MISMATCH"
    if b != e:
        bad += 1
    print(f"  {env:12s} begin={b:3d} end={e:3d}  {flag}")

print("\ntabular column counts:")
i = 0
while i < len(lines):
    m = re.search(r"\\begin\{tabular\}\{([^}]*)\}", lines[i])
    if not m:
        i += 1
        continue
    spec = re.sub(r"[^lcrp|@]", "", m.group(1))
    ncol = len(re.sub(r"[|@]", "", spec))
    start = i
    rows = []
    i += 1
    while i < len(lines) and "\\end{tabular}" not in lines[i]:
        ln = lines[i]
        if ("&" in ln and "\\\\" in ln
                and not ln.strip().startswith("%")
                and "cmidrule" not in ln and "multicolumn" not in ln):
            # count top-level & only; \& is an escaped ampersand
            n = len(re.findall(r"(?<!\\)&", ln)) + 1
            rows.append((i + 1, n))
        i += 1
    counts = {n for _, n in rows}
    ok = counts <= {ncol}
    print(f"  line {start+1}: spec={ncol} cols, data rows have {sorted(counts)}  "
          f"{'OK' if ok else 'MISMATCH'}")
    if not ok:
        bad += 1
        for ln, n in rows:
            if n != ncol:
                print(f"      line {ln}: {n} cells -- {lines[ln-1][:70]}")

print(f"\n{'PASS' if bad == 0 else str(bad) + ' PROBLEM(S)'}")
sys.exit(1 if bad else 0)
