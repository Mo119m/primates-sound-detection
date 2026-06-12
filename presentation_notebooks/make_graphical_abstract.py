"""
Generate the MethodsX graphical abstract for the manuscript.

A single landscape figure that summarizes the whole pipeline at a glance:
  reference clips -> mel-spectrogram + augmentation -> VGG19 + frequency-position
  CRNN classifier -> sliding-window detection + low-frequency gate ->
  three-filter automatic cleanup -> hard-negative mining (feedback loop).

MethodsX spec: min. 531 x 1328 px (h x w), readable at 5 x 13 cm, submitted as a
SEPARATE file (TIFF/EPS/PDF/MS Office). We emit both a vector PDF (best for
submission) and a high-resolution PNG (for quick preview / slides).

Outputs:
  figures/graphical_abstract.pdf
  figures/graphical_abstract.png

Pure matplotlib -- no graphviz dependency. Matches the style of
make_pipeline_figures.py.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = Path(__file__).parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── style ──────────────────────────────────────────────────────────────────
BOX_KW = dict(boxstyle="round,pad=0.4,rounding_size=0.6", linewidth=1.8, zorder=3)
ARROW_KW = dict(
    arrowstyle="-|>,head_length=7,head_width=4.5",
    linewidth=2.0, color="#34495E", zorder=2,
    connectionstyle="arc3,rad=0.0",
)
TITLE_SIZE = 9.0
BODY_SIZE = 7.0

# Aspect ratio ~2.5:1 (MethodsX wants h x w = 531 x 1328 px minimum).
fig, ax = plt.subplots(figsize=(13.28, 5.31), dpi=240)
ax.set_xlim(0, 100)
ax.set_ylim(0, 40)
ax.axis("off")

# Outer frame.
ax.add_patch(Rectangle((0.4, 0.4), 99.2, 39.2, fill=False,
                        edgecolor="#BDC3C7", linewidth=1.2, zorder=0))

# ── header ─────────────────────────────────────────────────────────────────
ax.text(50, 37.6,
        "Detecting primate vocalizations in long tropical-forest recordings",
        ha="center", va="center", fontsize=13.5, fontweight="bold",
        color="#1B2631")
ax.text(50, 34.6,
        "A VGG19 frequency-position CRNN with two-stage transfer learning, a "
        "hard-negative confuser class, and automatic false-positive cleanup",
        ha="center", va="center", fontsize=8.6, color="#5D6D7E", style="italic")

# ── pipeline boxes ─────────────────────────────────────────────────────────
# Six stages evenly spaced. Box width 14, gap 2.6 for arrows.
BW, BH, BY = 14.0, 14.0, 14.5
XS = [1.4, 17.6, 33.8, 50.0, 66.2, 82.4]

boxes = [
    dict(title="Reference clips",
         body="Cernic + Colobus\n+ confuser class\n+ Background\n(4 classes)",
         face="#F4F6F8", edge="#2C3E50"),
    dict(title="Mel-spectrogram\n+ augmentation",
         body="128 mel bins,\n20–8 000 Hz; 7× aug;\nHF-nuisance aug\nfor Colobus (V12)",
         face="#EBF5FB", edge="#2980B9"),
    dict(title="VGG19 + CRNN\nclassifier",
         body="frequency-position\nhead (CoordConv) →\nper-band Conv1D →\nBiLSTM → softmax",
         face="#E8F8F0", edge="#27AE60"),
    dict(title="Sliding-window\ndetection",
         body="2 s / 1 s windows,\ngroup → threshold →\nNMS → low-freq\ngate (Colobus)",
         face="#FDEBD0", edge="#E67E22"),
    dict(title="Three-filter\nauto-cleanup",
         body="Mahalanobis OOD\n+ YAMNet check\n+ temporal\nisolation",
         face="#FDEDEC", edge="#C0392B"),
    dict(title="Hard-negative\nmining",
         body="confirmed false\npositives folded\ninto Background\nfor retraining",
         face="#F4ECF7", edge="#8E44AD"),
]


def draw_box(x, b):
    ax.add_patch(FancyBboxPatch((x, BY), BW, BH,
                                facecolor=b["face"], edgecolor=b["edge"], **BOX_KW))
    n_title_lines = b["title"].count("\n") + 1
    ax.text(x + BW / 2, BY + BH - 1.4, b["title"],
            ha="center", va="top", fontsize=TITLE_SIZE, fontweight="bold",
            color=b["edge"], linespacing=1.2)
    ax.text(x + BW / 2, BY + BH - 2.0 - 1.9 * n_title_lines, b["body"],
            ha="center", va="top", fontsize=BODY_SIZE, color="#2C3E50",
            linespacing=1.45)


for x, b in zip(XS, boxes):
    draw_box(x, b)

# Forward arrows between consecutive stages.
for i in range(len(XS) - 1):
    x1 = XS[i] + BW + 0.3
    x2 = XS[i + 1] - 0.3
    ax.add_patch(FancyArrowPatch((x1, BY + BH / 2), (x2, BY + BH / 2), **ARROW_KW))

# ── feedback loop: cleanup/mining -> retraining ────────────────────────────
loop_kw = dict(
    arrowstyle="-|>,head_length=8,head_width=5",
    linewidth=2.0, color="#8E44AD", zorder=2,
    connectionstyle="arc3,rad=0.32", linestyle=(0, (6, 3)),
)
# From the bottom of the mining box back to the bottom of the reference/augment
# stage, arcing below the row.
start = (XS[5] + BW / 2, BY - 0.3)
end = (XS[0] + BW / 2, BY - 0.3)
ax.add_patch(FancyArrowPatch(start, end, **loop_kw))
ax.text(50, 5.6,
        "iterative hard-negative loop:  detect → clean → fold into Background → retrain",
        ha="center", va="center", fontsize=8.4, color="#8E44AD", fontweight="bold")

# ── footer: headline result ────────────────────────────────────────────────
ax.text(50, 1.9,
        "Two-stage transfer learning · 98.12 % validation accuracy · "
        "near-zero primate cross-confusion · automatic, near-zero manual listening",
        ha="center", va="center", fontsize=8.2, color="#34495E")

plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

pdf_path = OUT / "graphical_abstract.pdf"
png_path = OUT / "graphical_abstract.png"
fig.savefig(pdf_path, facecolor="white")
fig.savefig(png_path, dpi=240, facecolor="white")
print(f"Saved {pdf_path}")
print(f"Saved {png_path}")
