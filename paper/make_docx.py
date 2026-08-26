"""
Generate the Word (.docx) preview of the manuscript.

Word cannot display PDF (vector) figures inline, so a pandoc-embedded PDF
shows up as a blank box with the filename. The LaTeX source intentionally
uses the vector PDFs (best quality for the Overleaf/submission compile); for
the Word *preview* only, we swap \includegraphics{X.pdf} -> {X.png} on a
temporary copy so Word renders the figures.

Run from the paper/ directory:  python make_docx.py
"""
import re
import pathlib
import pypandoc

SRC = pathlib.Path("../overleaf/methodsx_manuscript.tex")
TMP = pathlib.Path("_manuscript_png.tex")
OUT = "methodsx_manuscript.docx"

text = SRC.read_text()
# Swap .pdf -> .png inside any \includegraphics{...} for the Word preview.
text = re.sub(r'(\\includegraphics(?:\[[^\]]*\])?\{[^}]+)\.pdf\}',
              r'\1.png}', text)
TMP.write_text(text)

# The reference doc only supplies fonts and heading styles, and it stopped
# being tracked on 2026-08-26 along with the .docx export it styled: a Word
# copy of the manuscript that sat fifteen days behind the .tex was doing more
# harm as an authoritative-looking stale artifact than the styling was doing
# good. So the conversion works without it rather than failing on its absence.
ref = pathlib.Path("reference_academic.docx")
extra = [f"--reference-doc={ref.name}"] if ref.exists() else []
if not extra:
    print("  no reference_academic.docx here; using pandoc's default styles")
try:
    pypandoc.convert_file(str(TMP), "docx", outputfile=OUT,
                          extra_args=extra)
    print(f"Wrote {OUT} (figures embedded as PNG so Word can display them)")
finally:
    TMP.unlink(missing_ok=True)
