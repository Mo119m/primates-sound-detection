"""
Generate two pipeline diagrams for the presentation:

  Figure A — Classification: model architecture (training on labelled clips)
  Figure B — Detection: sliding-window deployment on long field recordings

Outputs:
  figures/pipeline_classification.png
  figures/pipeline_detection.png

Pure matplotlib — no graphviz dependency.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── shared constants ──────────────────────────────────────────────────────

BOX_KW = dict(boxstyle="round,pad=0.45,rounding_size=0.5", linewidth=1.8, zorder=2)
TITLE_SIZE = 12
BODY_SIZE = 9.5
ARROW_KW = dict(
    arrowstyle="->,head_length=6,head_width=4.5",
    linewidth=1.6, color="#34495E", zorder=1,
    connectionstyle="arc3,rad=0.0",
)


def draw_box(ax, x, y, w, h, face, edge, title, body):
    ax.add_patch(FancyBboxPatch((x, y), w, h, facecolor=face, edgecolor=edge, **BOX_KW))
    ax.text(x + w / 2, y + h - 1.6, title,
            ha="center", va="top", fontsize=TITLE_SIZE, fontweight="bold", color=edge)
    ax.text(x + w / 2, y + h / 2 - 1.2, body,
            ha="center", va="center", fontsize=BODY_SIZE, color="#2C3E50", linespacing=1.35)


def draw_arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), **ARROW_KW))


# ══════════════════════════════════════════════════════════════════════════
# Figure A — Classification (model architecture)
# ══════════════════════════════════════════════════════════════════════════

fig_a, ax_a = plt.subplots(figsize=(14, 5.0), dpi=200)
ax_a.set_xlim(0, 100)
ax_a.set_ylim(0, 32)
ax_a.axis("off")

# Title
ax_a.text(50, 30.5,
          "Model architecture  —  training on labelled 5 s clips",
          ha="center", va="center", fontsize=14, fontweight="bold", color="#2C3E50")

# Boxes: evenly spaced across the width
boxes_a = [
    dict(x=1,  title="5 s clip",         body="labelled\nspecies audio",      face="#F4F6F8", edge="#2C3E50"),
    dict(x=18, title="Mel-spectrogram",   body="128 mel bins\n20–8 000 Hz",   face="#EBF5FB", edge="#2980B9"),
    dict(x=35, title="Resize",            body="→ 224×224\nRGB (3-ch stack)",  face="#EBF5FB", edge="#2980B9"),
    dict(x=52, title="VGG19 base",        body="ImageNet weights\n(frozen)\n~20 M params", face="#E8F8F0", edge="#27AE60"),
    dict(x=69, title="Custom head",       body="GAP → 512 → 256\nDropout 0.5",face="#FDEBD0", edge="#E67E22"),
    dict(x=86, title="Softmax",           body="4 classes:\n3 species\n+ Background", face="#F4ECF7", edge="#8E44AD"),
]
BW, BH = 13, 11
BY = 10
for b in boxes_a:
    draw_box(ax_a, b["x"], BY, BW, BH, b["face"], b["edge"], b["title"], b["body"])

# Arrows between consecutive boxes
for i in range(len(boxes_a) - 1):
    x1 = boxes_a[i]["x"] + BW + 0.2
    x2 = boxes_a[i + 1]["x"] - 0.2
    draw_arrow(ax_a, x1, BY + BH / 2, x2, BY + BH / 2)

# Footer
ax_a.text(50, 3.0,
          "Trained on ~870 labelled clips (×7 augmentation)  |  94.3 % validation accuracy",
          ha="center", va="center", fontsize=10, color="#7F8C8D", style="italic")

plt.tight_layout()
out_a = OUT / "pipeline_classification.png"
fig_a.savefig(out_a, bbox_inches="tight", dpi=220, facecolor="white")
print(f"Saved {out_a}")


# ══════════════════════════════════════════════════════════════════════════
# Figure B — Detection (sliding-window deployment)
# ══════════════════════════════════════════════════════════════════════════

fig_b, ax_b = plt.subplots(figsize=(14, 5.0), dpi=200)
ax_b.set_xlim(0, 100)
ax_b.set_ylim(0, 32)
ax_b.axis("off")

# Title
ax_b.text(50, 30.5,
          "Field deployment  —  sliding-window detection",
          ha="center", va="center", fontsize=14, fontweight="bold", color="#2C3E50")

boxes_b = [
    dict(x=1,  title="Long audio",       body="raw .wav\nfield recording\n(hours)",  face="#F4F6F8", edge="#2C3E50"),
    dict(x=18, title="Sliding window",    body="5 s window\n2.5 s stride\n(50 % overlap)", face="#FDEBD0", edge="#E67E22"),
    dict(x=35, title="Trained model",     body="mel-spec\n→ VGG19 + head\n(from Fig A)", face="#E8F8F0", edge="#27AE60"),
    dict(x=52, title="Per-window\nprediction", body="species +\nconfidence\nper window", face="#EBF5FB", edge="#2980B9"),
    dict(x=69, title="Threshold\n+ NMS",  body="conf ≥ 0.4\n+ overlap\nsuppression", face="#FDF2F8", edge="#C0392B"),
    dict(x=86, title="Detections",        body="species\nstart / end time\nconfidence",   face="#F4ECF7", edge="#8E44AD"),
]
for b in boxes_b:
    draw_box(ax_b, b["x"], BY, BW, BH, b["face"], b["edge"], b["title"], b["body"])

for i in range(len(boxes_b) - 1):
    x1 = boxes_b[i]["x"] + BW + 0.2
    x2 = boxes_b[i + 1]["x"] - 0.2
    draw_arrow(ax_b, x1, BY + BH / 2, x2, BY + BH / 2)

# Footer
ax_b.text(50, 3.0,
          "13 recordings from Makokou, Gabon (June 9, 2022)  |  ~10 000 sliding-window predictions",
          ha="center", va="center", fontsize=10, color="#7F8C8D", style="italic")

plt.tight_layout()
out_b = OUT / "pipeline_detection.png"
fig_b.savefig(out_b, bbox_inches="tight", dpi=220, facecolor="white")
print(f"Saved {out_b}")
