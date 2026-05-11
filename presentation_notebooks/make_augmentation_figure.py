"""
Build a data-augmentation pipeline diagram for the presentation.

Produces presentation_notebooks/figures/augmentation_pipeline.png.
Pure matplotlib — no graphviz dependency.

Layout: one "input clip" box on the left, a fan-out of 5 augmentation
operation boxes in the middle (each labelled with its multiplicity and
parameter range), and a "7 variants" summary box on the right.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


OUT = Path(__file__).parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)


# --- Figure config -------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 7.0), dpi=200)
ax.set_xlim(0, 100)
ax.set_ylim(0, 52)
ax.axis("off")


TITLE_SIZE = 12
BODY_SIZE = 9.5
BOX_KW = dict(boxstyle="round,pad=0.4,rounding_size=0.6",
              linewidth=1.6, zorder=2)


# Input box (left)
IN_X, IN_Y, IN_W, IN_H = 2, 22, 14, 8
ax.add_patch(FancyBboxPatch(
    (IN_X, IN_Y), IN_W, IN_H,
    facecolor="#F4F6F8", edgecolor="#2C3E50", **BOX_KW,
))
ax.text(IN_X + IN_W / 2, IN_Y + IN_H - 1.8, "1 species clip",
        ha="center", va="top", fontsize=TITLE_SIZE,
        fontweight="bold", color="#2C3E50")
ax.text(IN_X + IN_W / 2, IN_Y + IN_H / 2 - 1.2,
        "5 s audio\n→ mel-spec",
        ha="center", va="center", fontsize=BODY_SIZE,
        color="#2C3E50", linespacing=1.3)


# 5 augmentation operation boxes (middle)
OPS = [
    {
        "title": "Original",
        "mult": "× 1",
        "body": "pass through\nunchanged",
        "face": "#F4F6F8", "edge": "#2C3E50",
    },
    {
        "title": "Background mix",
        "mult": "× 3",
        "body": "SNR −5..10 dB\nrandom bg clip",
        "face": "#FDEBD0", "edge": "#E67E22",
    },
    {
        "title": "Time chop",
        "mult": "× 1",
        "body": "crop 10–30 %\nof time axis",
        "face": "#EBF5FB", "edge": "#2980B9",
    },
    {
        "title": "Frequency chop",
        "mult": "× 1",
        "body": "crop 10–30 %\nof mel axis",
        "face": "#E8F8F0", "edge": "#27AE60",
    },
    {
        "title": "Frequency shift",
        "mult": "× 1",
        "body": "translate\n±20 mel bins",
        "face": "#FDF2F8", "edge": "#C0392B",
    },
]

OPS_X = 32
OPS_W = 20
OPS_H = 7.0
# Evenly stack 5 boxes vertically from ~top of plot to bottom
OPS_YS = [42, 33, 24, 15, 6]  # top → bottom

for op, y in zip(OPS, OPS_YS):
    box = FancyBboxPatch(
        (OPS_X, y), OPS_W, OPS_H,
        facecolor=op["face"], edgecolor=op["edge"], **BOX_KW,
    )
    ax.add_patch(box)
    # Title (left-aligned in the box)
    ax.text(OPS_X + 1.0, y + OPS_H - 1.6,
            op["title"],
            ha="left", va="top", fontsize=TITLE_SIZE,
            fontweight="bold", color=op["edge"])
    # Multiplicity badge (right edge) — separate from title
    ax.text(OPS_X + OPS_W - 1.0, y + OPS_H - 1.6,
            op["mult"],
            ha="right", va="top", fontsize=11,
            fontweight="bold", color=op["edge"])
    # Body
    ax.text(OPS_X + OPS_W / 2, y + OPS_H / 2 - 1.4,
            op["body"],
            ha="center", va="center", fontsize=BODY_SIZE,
            color="#2C3E50", linespacing=1.3)


# Arrows from input → each op box
in_exit_x = IN_X + IN_W
in_exit_y = IN_Y + IN_H / 2
for y in OPS_YS:
    target_y = y + OPS_H / 2
    arrow = FancyArrowPatch(
        (in_exit_x + 0.2, in_exit_y),
        (OPS_X - 0.2, target_y),
        arrowstyle="->,head_length=5,head_width=4",
        linewidth=1.4, color="#34495E", zorder=1,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)


# Summary box (right)
SUM_X, SUM_Y, SUM_W, SUM_H = 84, 22, 14, 8
ax.add_patch(FancyBboxPatch(
    (SUM_X, SUM_Y), SUM_W, SUM_H,
    facecolor="#F4ECF7", edgecolor="#8E44AD", **BOX_KW,
))
ax.text(SUM_X + SUM_W / 2, SUM_Y + SUM_H - 1.8, "7 variants",
        ha="center", va="top", fontsize=TITLE_SIZE,
        fontweight="bold", color="#8E44AD")
ax.text(SUM_X + SUM_W / 2, SUM_Y + SUM_H / 2 - 1.2,
        "per input clip\n(× 7 multiplier)",
        ha="center", va="center", fontsize=BODY_SIZE,
        color="#2C3E50", linespacing=1.3)


# Arrows from each op box → summary
sum_entry_x = SUM_X
sum_entry_y = SUM_Y + SUM_H / 2
for y in OPS_YS:
    source_y = y + OPS_H / 2
    arrow = FancyArrowPatch(
        (OPS_X + OPS_W + 0.2, source_y),
        (sum_entry_x - 0.2, sum_entry_y),
        arrowstyle="->,head_length=5,head_width=4",
        linewidth=1.4, color="#34495E", zorder=1,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)


# Title + footer
ax.text(50, 50.5,
        "Data augmentation  —  × 7 effective training-set multiplier",
        ha="center", va="center", fontsize=14, fontweight="bold",
        color="#2C3E50")

ax.text(50, 1.5,
        "~870 species clips  →  ~6 000 effective training samples  |  "
        "teaches the model to be invariant to background, timing, and pitch",
        ha="center", va="center", fontsize=10,
        color="#7F8C8D", style="italic")


plt.tight_layout()
out_path = OUT / "augmentation_pipeline.png"
fig.savefig(out_path, bbox_inches="tight", dpi=220, facecolor="white")
print(f"Saved {out_path}")
