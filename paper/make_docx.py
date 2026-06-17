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

SRC = pathlib.Path("methodsx_manuscript.tex")
TMP = pathlib.Path("_manuscript_png.tex")
OUT = "methodsx_manuscript.docx"

text = SRC.read_text()
# Swap .pdf -> .png inside any \includegraphics{...} for the Word preview.
text = re.sub(r'(\\includegraphics(?:\[[^\]]*\])?\{[^}]+)\.pdf\}',
              r'\1.png}', text)
TMP.write_text(text)
try:
    pypandoc.convert_file(str(TMP), "docx", outputfile=OUT,
                          extra_args=["--reference-doc=reference_academic.docx"])
    print(f"Wrote {OUT} (figures embedded as PNG so Word can display them)")
finally:
    TMP.unlink(missing_ok=True)
