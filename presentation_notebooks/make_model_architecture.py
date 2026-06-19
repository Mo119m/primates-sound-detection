"""
Model-architecture diagram for the manuscript (V12).

Restrained near-monochrome journal style (single serif typeface, one ink
colour, hairline frames, neutral fills, Unicode glyphs -- no mathtext),
matching the other manuscript figures. Depicts the full classifier: the VGG19
backbone, the frequency-position (CoordConv) encoding, the four-band CRNN head,
the bidirectional LSTM, and the dense/softmax classifier. Output tensor shapes
are annotated on the right; the two novel components are flagged in italics.

Outputs:
  figures/model_architecture_v12.pdf
  figures/model_architecture_v12.png
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
mpl.rcParams["pdf.fonttype"] = 42  # embed TrueType (avoid Type 3; Elsevier requirement)
mpl.rcParams["ps.fonttype"] = 42

OUT = Path(__file__).parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#222222"
FRAME = "#8C8C8C"
FILL = "#F0EFEC"
SHAPE = "#555555"

fig, ax = plt.subplots(figsize=(7.2, 9.2), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
fig.patch.set_facecolor("white")

ax.text(50, 97.5, "Model architecture", ha="center", va="center",
        fontsize=14, color=INK)

CX, BW = 46, 50          # box centre-x and width
LX = CX - BW / 2         # left edge
SHX = CX + BW / 2 + 2    # x for shape annotation (right of box)


def box(cy, h, lines, shape=None, note=None, fill=FILL):
    ax.add_patch(Rectangle((LX, cy - h / 2), BW, h, facecolor=fill,
                           edgecolor=FRAME, linewidth=0.9, zorder=3))
    ax.text(CX, cy, lines, ha="center", va="center", fontsize=8.6,
            color=INK, linespacing=1.3, zorder=4)
    if shape:
        ax.text(SHX, cy, shape, ha="left", va="center", fontsize=7.6,
                color=SHAPE, family="monospace")
    if note:
        ax.text(LX - 2, cy, note, ha="right", va="center", fontsize=7.6,
                color=INK, style="italic", linespacing=1.1)


def varrow(y1, y2):
    ax.add_patch(FancyArrowPatch((CX, y1), (CX, y2),
                 arrowstyle="-|>,head_length=5,head_width=3",
                 color=INK, linewidth=1.0, zorder=2))


# ── main vertical stack ────────────────────────────────────────────────────
box(90.5, 6.0, "Mel-spectrogram input", shape="224×224×3")
varrow(87.5, 85.0)
box(81.5, 7.0, "VGG19 backbone — block4_conv4\n(ImageNet-pretrained)",
    shape="28×28×512")
varrow(78.0, 75.3)
box(71.0, 7.6,
    "Frequency-coordinate channel (CoordConv)\nConv2D(128, 3) + BN + ReLU",
    shape="28×28×128", note="frequency-\nposition\nencoding")

# ── four-band split ────────────────────────────────────────────────────────
ax.text(CX, 63.2, "split mel axis into 4 bands · average-pool frequency",
        ha="center", va="center", fontsize=7.6, color=INK, style="italic")
band_y = 57.0
band_h = 5.6
band_w = 9.5
band_cxs = [LX + 7 + i * 12.2 for i in range(4)]
for i, bx in enumerate(band_cxs):
    ax.add_patch(Rectangle((bx - band_w / 2, band_y - band_h / 2), band_w,
                           band_h, facecolor="white", edgecolor=FRAME,
                           linewidth=0.8, zorder=3))
    ax.text(bx, band_y, f"band {i+1}", ha="center", va="center",
            fontsize=6.8, color=INK)
    # fan-out from CoordConv box, fan-in to concat box
    ax.add_patch(FancyArrowPatch((CX, 67.2), (bx, band_y + band_h / 2),
                 arrowstyle="-|>,head_length=3.5,head_width=2.4",
                 color=INK, linewidth=0.7, zorder=2,
                 connectionstyle="arc3,rad=0.0"))
    ax.add_patch(FancyArrowPatch((bx, band_y - band_h / 2), (CX, 50.3),
                 arrowstyle="-|>,head_length=3.5,head_width=2.4",
                 color=INK, linewidth=0.7, zorder=2,
                 connectionstyle="arc3,rad=0.0"))
ax.text(SHX, band_y, "4 × (28×128)", ha="left", va="center", fontsize=7.6,
        color=SHAPE, family="monospace")
ax.text(LX - 2, band_y, "per-band\nConv1D(128, 3)", ha="right", va="center",
        fontsize=7.0, color=INK, style="italic", linespacing=1.1)

box(46.5, 7.0, "Concatenate + cross-band\nConv1D(256, 3) + BN + ReLU",
    shape="28×512 → 28×256")
varrow(43.0, 40.3)
box(36.5, 7.0, "Bidirectional LSTM (128 × 2)\ninput dropout 0.3",
    shape="28×256")
varrow(33.0, 30.3)
box(27.0, 6.0, "Global max-pool + average-pool", shape="512")
varrow(24.0, 21.3)
box(17.5, 7.0, "Dense(512) → Dense(256)\nReLU, dropout 0.5", shape="256")
varrow(14.0, 11.3)
box(7.0, 7.2, "Softmax — 4 classes\nCernic · Colobus · confuser · Background",
    shape="4")

plt.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.01)

pdf_path = OUT / "model_architecture_v12.pdf"
png_path = OUT / "model_architecture_v12.png"
fig.savefig(pdf_path, facecolor="white", bbox_inches="tight")
fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")
print(f"Saved {pdf_path}")
print(f"Saved {png_path}")
