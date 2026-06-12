"""
Generate the MethodsX graphical abstract for the manuscript.

A single landscape figure that summarizes the whole pipeline at a glance:
  reference clips -> mel-spectrogram + augmentation -> VGG19 + frequency-position
  CRNN classifier -> sliding-window detection + low-frequency gate ->
  three-filter automatic cleanup -> hard-negative mining (feedback loop).

Design: restrained, journal-style typography in a Times-compatible serif
(Liberation Serif -- the metric-identical open substitute for Times New Roman,
which is proprietary and not installable here). Muted, desaturated accent
palette; hairline rules; generous whitespace. No saturated "flowchart" fills.

MethodsX spec: min. 531 x 1328 px (h x w), readable at 5 x 13 cm, submitted as a
SEPARATE file (TIFF/EPS/PDF/MS Office). We emit both a vector PDF (best for
submission) and a high-resolution PNG (for quick preview / slides).

Outputs:
  figures/graphical_abstract.pdf
  figures/graphical_abstract.png
"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

# ── typography: Times-compatible serif ─────────────────────────────────────
# Prefer real Times New Roman if present, else Liberation Serif / Nimbus / STIX.
_SERIF_PREF = ["Times New Roman", "Liberation Serif", "Nimbus Roman",
               "STIXGeneral", "DejaVu Serif"]
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = _SERIF_PREF
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── palette (muted, desaturated; cool -> warm -> plum) ─────────────────────
INK = "#1E2A32"        # near-black serif text
SUBINK = "#5A6670"     # muted captions
BODY = "#33373A"       # box body text
PAPER = "#FBFAF7"      # warm off-white box fill
FRAME = "#CBC7BF"      # hairline frame / borders
ARROW = "#8C887F"      # connector grey
LOOP = "#6E5B72"       # feedback-loop plum

ACCENTS = ["#3E5C76", "#4F7C82", "#5B7553", "#8C6D46", "#99584B", "#6E5B72"]
NUMERALS = ["I", "II", "III", "IV", "V", "VI"]

# Aspect ratio ~2.5:1 (MethodsX wants h x w = 531 x 1328 px minimum).
fig, ax = plt.subplots(figsize=(13.28, 5.31), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 40)
ax.axis("off")
fig.patch.set_facecolor("white")

# Outer hairline frame.
ax.add_patch(Rectangle((0.5, 0.5), 99.0, 39.0, fill=False,
                        edgecolor=FRAME, linewidth=1.0, zorder=0))

# ── header ─────────────────────────────────────────────────────────────────
ax.text(50, 37.0,
        "Detecting primate vocalizations in long tropical-forest recordings",
        ha="center", va="center", fontsize=15.5, color=INK)
ax.text(50, 33.9,
        "A transfer-learning classifier with an automatic false-positive "
        "cleanup loop",
        ha="center", va="center", fontsize=10.0, color=SUBINK, style="italic")
# Hairline rule under the header.
ax.plot([10, 90], [31.6, 31.6], color=FRAME, linewidth=0.9, zorder=1)

# ── pipeline boxes ─────────────────────────────────────────────────────────
# Six stages evenly spaced. Box width 14.4, gap ~1.9 for connectors.
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

BOX_KW = dict(boxstyle="round,pad=0.32,rounding_size=0.22",
              linewidth=1.0, zorder=3)


def draw_box(x, b, accent, numeral):
    # Card.
    ax.add_patch(FancyBboxPatch((x, BY), BW, BH,
                                facecolor=PAPER, edgecolor=FRAME, **BOX_KW))
    # Thin accent rule along the top edge.
    ax.plot([x + 0.7, x + BW - 0.7], [BY + BH - 0.55, BY + BH - 0.55],
            color=accent, linewidth=2.4, zorder=4, solid_capstyle="round")
    # Small roman numeral, in the accent colour, top-left.
    ax.text(x + 1.0, BY + BH - 2.0, numeral,
            ha="left", va="top", fontsize=8.5, color=accent, style="italic")
    # Stage title (serif, accent colour).
    ax.text(x + BW / 2, BY + BH - 2.0, b["title"],
            ha="center", va="top", fontsize=11.5, color=accent,
            linespacing=1.15)
    # Body, vertically centred in the lower portion of the card.
    ax.text(x + BW / 2, BY + BH * 0.36, b["body"],
            ha="center", va="center", fontsize=7.6, color=BODY, linespacing=1.55)


for x, b, a, num in zip(XS, boxes, ACCENTS, NUMERALS):
    draw_box(x, b, a, num)

# Slim connectors between consecutive stages.
arrow_kw = dict(arrowstyle="-|>,head_length=5,head_width=3.0",
                linewidth=1.1, color=ARROW, zorder=2,
                connectionstyle="arc3,rad=0.0")
for i in range(len(XS) - 1):
    x1 = XS[i] + BW + 0.25
    x2 = XS[i + 1] - 0.25
    ax.add_patch(FancyArrowPatch((x1, BY + BH / 2), (x2, BY + BH / 2), **arrow_kw))

# ── feedback loop: mining -> training, arcing below the row ─────────────────
loop_kw = dict(arrowstyle="-|>,head_length=6.0,head_width=3.6",
               linewidth=1.3, color=LOOP, zorder=2,
               connectionstyle="bar,fraction=-0.05", linestyle=(0, (5, 2.6)))
ax.add_patch(FancyArrowPatch((XS[5] + BW / 2, BY - 0.3),
                             (XS[0] + BW / 2, BY - 0.3), **loop_kw))
ax.text(50, 5.6,
        "iterative loop · detect, clean, fold into Background, retrain",
        ha="center", va="center", fontsize=9.0, color=LOOP, style="italic")

# ── footer: headline result ────────────────────────────────────────────────
ax.plot([18, 82], [3.1, 3.1], color=FRAME, linewidth=0.8, zorder=1)
ax.text(50, 1.85,
        "98.12 % validation accuracy · near-zero confusion between "
        "the two primate classes · minimal manual listening",
        ha="center", va="center", fontsize=8.4, color=SUBINK)

plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

pdf_path = OUT / "graphical_abstract.pdf"
png_path = OUT / "graphical_abstract.png"
fig.savefig(pdf_path, facecolor="white")
fig.savefig(png_path, dpi=300, facecolor="white")
print(f"Saved {pdf_path}")
print(f"Saved {png_path}")
