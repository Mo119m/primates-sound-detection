"""
Build a clean pipeline architecture diagram for the presentation.

Produces presentation_notebooks/figures/pipeline_architecture.png.
Pure matplotlib — no graphviz dependency.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


OUT = Path(__file__).parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# --- Figure config -------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 4.2), dpi=200)
ax.set_xlim(0, 100)
ax.set_ylim(0, 30)
ax.axis("off")


BOX_H = 11
BOX_Y = 10  # bottom of each box

# Style
TITLE_SIZE = 12
BODY_SIZE = 10
BOX_KW = dict(boxstyle="round,pad=0.5,rounding_size=0.8",
              linewidth=1.6, zorder=2)

# Per-box colors
STAGES = [
    {
        "x": 2,  "w": 14, "title": "Long audio",
        "body": "raw .wav\nfield recording",
        "face": "#F4F6F8", "edge": "#2C3E50",
    },
    {
        "x": 19, "w": 14, "title": "Sliding window",
        "body": "5 s window\n2.5 s stride",
        "face": "#FDEBD0", "edge": "#E67E22",
    },
    {
        "x": 36, "w": 14, "title": "Mel-spectrogram",
        "body": "128 mel bins\n20–8000 Hz\n→ 224×224 RGB",
        "face": "#EBF5FB", "edge": "#2980B9",
    },
    {
        "x": 53, "w": 14, "title": "VGG19 base",
        "body": "ImageNet weights\n(frozen)\n~20M params",
        "face": "#E8F8F0", "edge": "#27AE60",
    },
    {
        "x": 70, "w": 14, "title": "Custom head",
        "body": "GAP → 512 → 256\n+ Dropout 0.5",
        "face": "#FDF2F8", "edge": "#C0392B",
    },
    {
        "x": 87, "w": 11, "title": "Softmax",
        "body": "4 classes:\n3 species\n+ Background",
        "face": "#F4ECF7", "edge": "#8E44AD",
    },
]

# Draw boxes
for s in STAGES:
    box = FancyBboxPatch(
        (s["x"], BOX_Y), s["w"], BOX_H,
        facecolor=s["face"], edgecolor=s["edge"], **BOX_KW,
    )
    ax.add_patch(box)
    cx = s["x"] + s["w"] / 2
    ax.text(cx, BOX_Y + BOX_H - 2.2, s["title"],
            ha="center", va="top", fontsize=TITLE_SIZE,
            fontweight="bold", color=s["edge"])
    ax.text(cx, BOX_Y + BOX_H / 2 - 1.2, s["body"],
            ha="center", va="center", fontsize=BODY_SIZE,
            color="#2C3E50", linespacing=1.3)


# Draw arrows between boxes
for a, b in zip(STAGES[:-1], STAGES[1:]):
    x1 = a["x"] + a["w"]
    x2 = b["x"]
    y = BOX_Y + BOX_H / 2
    arrow = FancyArrowPatch(
        (x1 + 0.2, y), (x2 - 0.2, y),
        arrowstyle="->,head_length=6,head_width=5",
        linewidth=1.8, color="#34495E", zorder=1,
    )
    ax.add_patch(arrow)


# Upper/lower annotation bands
ax.text(50, 27,
        "Audio-as-image classification via transfer learning",
        ha="center", va="center", fontsize=14, fontweight="bold",
        color="#2C3E50")

ax.text(50, 3.5,
        "Trained on ~870 labelled clips (× 7 augmentation)  |  "
        "94.3% validation accuracy  |  "
        "Deployed on multi-hour Makokou field recordings",
        ha="center", va="center", fontsize=10,
        color="#7F8C8D", style="italic")


plt.tight_layout()
out_path = OUT / "pipeline_architecture.png"
fig.savefig(out_path, bbox_inches="tight", dpi=220, facecolor="white")
print(f"Saved {out_path}")
