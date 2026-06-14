"""
Generate a confusion-matrix heatmap for the V12 validation results.

Uses the exact values from the V12 training output (3717 clips total).
Style matches the graphical abstract: Liberation Serif, muted academic palette.

Outputs:
  figures/confusion_matrix_v12.pdf
  figures/confusion_matrix_v12.png
"""
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── typography ────────────────────────────────────────────────────────────
_SERIF_PREF = ["Times New Roman", "Liberation Serif", "Nimbus Roman",
               "STIXGeneral", "DejaVu Serif"]
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = _SERIF_PREF
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── V12 exact confusion matrix ───────────────────────────────────────────
CLASSES = ["Cernic", "Colobus\nguereza", "Colobus\nconfuser", "Background"]
CM = np.array([
    [498,    0,    9,   11],
    [  0, 1100,    5,    6],
    [  5,    2,  895,   13],
    [  8,    4,    7, 1154],
])

# ── palette: warm off-white → desaturated teal ───────────────────────────
CMAP = LinearSegmentedColormap.from_list(
    "academic", ["#FBFAF7", "#D6E8E4", "#89B8AD", "#3E7E6D", "#1E4D40"])

INK = "#1E2A32"
SUBINK = "#5A6670"

# ── figure ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.8), dpi=300)

# Normalise rows to percentages for colour mapping
row_sums = CM.sum(axis=1, keepdims=True)
CM_pct = CM / row_sums * 100

im = ax.imshow(CM_pct, cmap=CMAP, aspect="equal", vmin=0, vmax=100)

# Annotate each cell: count on top, percentage below
for i in range(4):
    for j in range(4):
        count = CM[i, j]
        pct = CM_pct[i, j]
        color = "white" if pct > 60 else INK
        if i == j:
            ax.text(j, i - 0.12, f"{count}", ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color)
            ax.text(j, i + 0.22, f"({pct:.1f}%)", ha="center", va="center",
                    fontsize=8.5, color=color)
        else:
            ax.text(j, i, f"{count}", ha="center", va="center",
                    fontsize=11, color=color)

ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(CLASSES, fontsize=9.5, color=INK)
ax.set_yticklabels(CLASSES, fontsize=9.5, color=INK)
ax.set_xlabel("Predicted class", fontsize=11, color=INK, labelpad=8)
ax.set_ylabel("True class", fontsize=11, color=INK, labelpad=8)

ax.set_title("V12 validation confusion matrix  (n = 3 717,  accuracy = 98.12%)",
             fontsize=11.5, color=INK, pad=12)

# Thin border
for spine in ax.spines.values():
    spine.set_color("#CBC7BF")
    spine.set_linewidth(0.8)

ax.tick_params(length=0)

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Row-normalised (%)", fontsize=9, color=SUBINK)
cbar.ax.tick_params(labelsize=8, colors=SUBINK)
cbar.outline.set_edgecolor("#CBC7BF")
cbar.outline.set_linewidth(0.8)

plt.tight_layout()

pdf_path = OUT / "confusion_matrix_v12.pdf"
png_path = OUT / "confusion_matrix_v12.png"
fig.savefig(pdf_path, facecolor="white", bbox_inches="tight")
fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")
print(f"Saved {pdf_path}")
print(f"Saved {png_path}")
