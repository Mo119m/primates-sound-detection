"""
Generate the MethodsX graphical abstract.

Restrained, near-monochrome journal style consistent with the manuscript's
other figures: one serif typeface (Liberation Serif, a Times clone), a single
ink colour plus one grey for secondary text, thin hairline frames, white
fills, and minimal ornament -- no coloured "flowchart" accents.

A single landscape figure summarising the pipeline at a glance:
  reference clips -> spectrogram + augmentation -> VGG19 frequency-position
  CRNN classifier -> sliding-window detection + low-frequency gate ->
  three-filter automatic cleanup -> hard-negative mining (feedback loop).

MethodsX spec: min. 531 x 1328 px (h x w), readable at 5 x 13 cm, submitted as
a SEPARATE file. Emits a vector PDF and a high-resolution PNG.

Outputs:
  figures/graphical_abstract.pdf
  figures/graphical_abstract.png
"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

_SERIF_PREF = ["Times New Roman", "Liberation Serif", "Nimbus Roman",
               "STIXGeneral", "DejaVu Serif"]
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = _SERIF_PREF
mpl.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── near-monochrome palette ────────────────────────────────────────────────
INK = "#222222"        # titles, body, arrows
SUBINK = "#5A6670"     # numerals, captions, footer
FRAME = "#8C8C8C"      # hairline frames

NUMERALS = ["I", "II", "III", "IV", "V", "VI"]

# Aspect ratio ~2.5:1 (MethodsX min. h x w = 531 x 1328 px).
fig, ax = plt.subplots(figsize=(13.28, 5.31), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 40)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── header ─────────────────────────────────────────────────────────────────
ax.text(50, 37.0,
        "Transfer-learning detection of primate vocalizations in "
        "tropical-forest recordings",
        ha="center", va="center", fontsize=14.5, color=INK)
ax.text(50, 33.9,
        "with an automatic false-positive cleanup loop",
        ha="center", va="center", fontsize=10.0, color=SUBINK, style="italic")
ax.plot([10, 90], [31.6, 31.6], color=FRAME, linewidth=0.8, zorder=1)

# ── pipeline boxes ─────────────────────────────────────────────────────────
BW, BH, BY = 14.4, 15.6, 11.6
XS = [1.6, 17.9, 34.2, 50.5, 66.8, 83.1]

boxes = [
    dict(title="Reference clips",
         body="Curated calls in four\nclasses: Cernic, Colobus,\na hard-negative confuser,\nand Background"),
    dict(title="Spectrogram\n& augmentation",
         body="128-mel spectrograms;\n7× augmentation, plus a\nhigh-frequency nuisance\nvariant for Colobus"),
    dict(title="Classifier",
         body="VGG19 backbone with a\nfrequency-position CRNN\nhead; two-stage\ntransfer learning"),
    dict(title="Sliding-window\ndetection",
         body="2 s windows; grouping,\nthresholding, NMS, and a\nlow-frequency energy\ngate for Colobus"),
    dict(title="Automatic\ncleanup",
         body="Three independent filters:\nMahalanobis distance,\nYAMNet tagging, and\ntemporal isolation"),
    dict(title="Hard-negative\nmining",
         body="Confirmed false positives\nrecycled into Background\nfor iterative\nretraining"),
]


def draw_box(x, b, numeral):
    ax.add_patch(Rectangle((x, BY), BW, BH, facecolor="white",
                           edgecolor=FRAME, linewidth=0.9, zorder=3))
    # small roman numeral, top-left, grey
    ax.text(x + 1.0, BY + BH - 1.8, numeral, ha="left", va="top",
            fontsize=8.0, color=SUBINK, style="italic")
    # stage title, ink
    ax.text(x + BW / 2, BY + BH - 1.8, b["title"], ha="center", va="top",
            fontsize=11.0, color=INK, linespacing=1.15)
    # body, ink
    ax.text(x + BW / 2, BY + BH * 0.34, b["body"], ha="center", va="center",
            fontsize=7.5, color=INK, linespacing=1.55)


for x, b, num in zip(XS, boxes, NUMERALS):
    draw_box(x, b, num)

# slim connectors between consecutive stages
arrow_kw = dict(arrowstyle="-|>,head_length=5,head_width=3.0",
                linewidth=1.0, color=INK, zorder=2,
                connectionstyle="arc3,rad=0.0")
for i in range(len(XS) - 1):
    x1 = XS[i] + BW + 0.25
    x2 = XS[i + 1] - 0.25
    ax.add_patch(FancyArrowPatch((x1, BY + BH / 2), (x2, BY + BH / 2), **arrow_kw))

# ── feedback loop: mining -> reference, arcing below the row ────────────────
loop_kw = dict(arrowstyle="-|>,head_length=6.0,head_width=3.6",
               linewidth=1.1, color=INK, zorder=2,
               connectionstyle="bar,fraction=-0.05", linestyle=(0, (5, 2.6)))
ax.add_patch(FancyArrowPatch((XS[5] + BW / 2, BY - 0.3),
                             (XS[0] + BW / 2, BY - 0.3), **loop_kw))
ax.text(50, 5.6,
        "iterative loop · detect, clean, fold into Background, retrain",
        ha="center", va="center", fontsize=9.0, color=SUBINK, style="italic")

# ── footer: headline result ────────────────────────────────────────────────
ax.plot([18, 82], [3.1, 3.1], color=FRAME, linewidth=0.8, zorder=1)
ax.text(50, 1.85,
        "98.12 % validation accuracy · zero confusion between "
        "the two primate species · minimal manual listening",
        ha="center", va="center", fontsize=8.4, color=SUBINK)

plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

pdf_path = OUT / "graphical_abstract.pdf"
png_path = OUT / "graphical_abstract.png"
fig.savefig(pdf_path, facecolor="white")
fig.savefig(png_path, dpi=300, facecolor="white")
print(f"Saved {pdf_path}")
print(f"Saved {png_path}")
