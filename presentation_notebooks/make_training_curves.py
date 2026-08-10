"""
Generate the V12 two-stage training curves (accuracy and loss).

Per-epoch values are taken directly from the recorded V12 training log:
  Stage 1 -- frozen VGG19 base, head training (29 epochs, early stopping).
  Stage 2 -- last two VGG19 blocks unfrozen, fine-tuning (28 epochs).
The deployed model (best_model_v12.h5) is the best-val-accuracy checkpoint,
stage-2 epoch 24.

The checkpoint is marked on the plot but its validation accuracy is NOT printed.
That figure was withdrawn from the manuscript: the split behind it was taken
after augmentation, so roughly 79 % of source clips placed a near-duplicate on
both sides and the number measures memorisation. The curves themselves are kept
because they document the two-stage training procedure; the caption says
explicitly that the validation trace is optimistic throughout.

Typography comes from _figstyle, and the figure is drawn at the width it will
occupy on the page, so a point here is a point in the PDF -- no scaling, and
nothing to compensate for. Do not add a local fontsize= anywhere in this file.

Outputs:
  figures/training_curves_v12.pdf
  figures/training_curves_v12.png
"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

import _figstyle
_figstyle.apply()

OUT = Path(__file__).parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── per-epoch metrics from the training log ────────────────────────────────
# Stage 1: frozen base, head training.
s1_acc = [0.7061, 0.9034, 0.9320, 0.9479, 0.9534, 0.9593, 0.9665, 0.9698,
          0.9757, 0.9765, 0.9796, 0.9815, 0.9818, 0.9835, 0.9860, 0.9935,
          0.9946, 0.9932, 0.9946, 0.9950, 0.9936, 0.9952, 0.9954, 0.9957,
          0.9968, 0.9983, 0.9970, 0.9985, 0.9980]
s1_val = [0.8752, 0.8953, 0.9481, 0.9389, 0.9545, 0.9567, 0.9535, 0.9518,
          0.9524, 0.9685, 0.9653, 0.9545, 0.9672, 0.9650, 0.9618, 0.9715,
          0.9688, 0.9669, 0.9718, 0.9728, 0.9701, 0.9650, 0.9607, 0.9669,
          0.9701, 0.9723, 0.9742, 0.9752, 0.9691]
s1_loss = [0.7051, 0.2952, 0.2057, 0.1643, 0.1369, 0.1184, 0.0987, 0.0886,
           0.0757, 0.0637, 0.0612, 0.0513, 0.0489, 0.0506, 0.0395, 0.0197,
           0.0165, 0.0179, 0.0165, 0.0128, 0.0177, 0.0118, 0.0124, 0.0148,
           0.0089, 0.0047, 0.0075, 0.0050, 0.0056]
s1_vloss = [0.3542, 0.3063, 0.1665, 0.1842, 0.1404, 0.1344, 0.1542, 0.1418,
            0.1704, 0.1193, 0.1241, 0.1784, 0.1352, 0.1255, 0.1643, 0.1201,
            0.1408, 0.1496, 0.1168, 0.1250, 0.1373, 0.1567, 0.1873, 0.1535,
            0.1462, 0.1400, 0.1506, 0.1348, 0.1742]

# Stage 2: last two VGG19 blocks unfrozen, fine-tuning.
s2_acc = [0.9784, 0.9868, 0.9874, 0.9874, 0.9894, 0.9896, 0.9926, 0.9926,
          0.9935, 0.9883, 0.9936, 0.9907, 0.9923, 0.9954, 0.9943, 0.9929,
          0.9977, 0.9987, 0.9984, 0.9981, 0.9988, 0.9973, 0.9987, 0.9989,
          0.9988, 0.9986, 0.9988, 0.9998]
s2_val = [0.9524, 0.9634, 0.9583, 0.9631, 0.9648, 0.9650, 0.9384, 0.9672,
          0.9572, 0.9572, 0.9726, 0.9527, 0.9556, 0.9683, 0.9623, 0.9306,
          0.9777, 0.9787, 0.9763, 0.9771, 0.9782, 0.9680, 0.9731, 0.9812,
          0.9787, 0.9774, 0.9712, 0.9798]
s2_loss = [0.0668, 0.0413, 0.0363, 0.0369, 0.0305, 0.0305, 0.0228, 0.0204,
           0.0190, 0.0332, 0.0189, 0.0255, 0.0217, 0.0143, 0.0165, 0.0212,
           0.0076, 0.0052, 0.0051, 0.0059, 0.0034, 0.0083, 0.0045, 0.0032,
           0.0030, 0.0042, 0.0036, 0.0013]
s2_vloss = [0.2204, 0.1560, 0.1552, 0.1453, 0.1697, 0.1458, 0.3128, 0.1201,
            0.1836, 0.1656, 0.1116, 0.2136, 0.2110, 0.1451, 0.1357, 0.3402,
            0.1012, 0.0973, 0.0975, 0.1045, 0.1104, 0.1643, 0.1314, 0.0994,
            0.1060, 0.1125, 0.1457, 0.1049]

n1, n2 = len(s1_acc), len(s2_acc)
x1 = list(range(1, n1 + 1))
x2 = list(range(n1 + 1, n1 + n2 + 1))
boundary = n1 + 0.5
# Deployed checkpoint: stage-2 epoch 24 (global epoch n1 + 24).
best_global = n1 + 24
best_val = 0.9812

TRAIN = _figstyle.TRAIN
VAL = _figstyle.VAL
INK = _figstyle.INK
SUBINK = _figstyle.SUBINK
FRAME = _figstyle.FRAME

fig, (axA, axL) = plt.subplots(1, 2, figsize=(_figstyle.TEXT_WIDTH_IN, 2.9), dpi=300)


def style(ax):
    for s in ax.spines.values():
        s.set_color(FRAME)
        s.set_linewidth(0.8)
    ax.tick_params(length=3, colors=SUBINK)
    ax.grid(True, color=_figstyle.GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.axvline(boundary, color=SUBINK, linestyle=(0, (4, 3)), linewidth=1.0,
               zorder=1)


# ── accuracy panel ─────────────────────────────────────────────────────────
axA.plot(x1, s1_acc, color=TRAIN, linewidth=1.6, label="Training")
axA.plot(x2, s2_acc, color=TRAIN, linewidth=1.6)
axA.plot(x1, s1_val, color=VAL, linewidth=1.6, label="Validation")
axA.plot(x2, s2_val, color=VAL, linewidth=1.6)
# The best-validation checkpoint is marked, but its VALUE is deliberately not
# printed. That number (98.12 %) has been withdrawn from the manuscript: the
# split it comes from was taken after augmentation, so ~79 % of source clips
# put a near-duplicate on both sides and the figure would be advertising a
# measurement of memorisation. Leaving it on the plot while the text explains
# why it was retracted is the kind of inconsistency a reviewer notices first.
axA.scatter([best_global], [best_val], s=42, facecolor="white",
            edgecolor=VAL, linewidth=1.6, zorder=5)
# No in-plot label. A leader line long enough to reach clear space cuts
# diagonally across the data, and the caption has room to say what the marker
# is. One mark, no annotation, explanation underneath.
style(axA)
axA.set_xlabel("Epoch", color=INK)
axA.set_ylabel("Accuracy", color=INK)
axA.set_ylim(0.68, 1.005)
axA.legend(loc="lower right", frameon=False)

# ── loss panel ───────────────────────────────────────────────────────────
axL.plot(x1, s1_loss, color=TRAIN, linewidth=1.6, label="Training")
axL.plot(x2, s2_loss, color=TRAIN, linewidth=1.6)
axL.plot(x1, s1_vloss, color=VAL, linewidth=1.6, label="Validation")
axL.plot(x2, s2_vloss, color=VAL, linewidth=1.6)
style(axL)
axL.set_xlabel("Epoch", color=INK)
axL.set_ylabel("Loss", color=INK)
axL.legend(loc="upper right", frameon=False)

# Stage labels (placed above each panel region).
for ax in (axA, axL):
    ymax = ax.get_ylim()[1]
    ax.text(n1 / 2, ymax, "Stage 1", ha="center", va="bottom", color=SUBINK)
    ax.text(n1 + n2 / 2, ymax, "Stage 2", ha="center", va="bottom",
            color=SUBINK)

fig.suptitle("V12 two-stage training history", color=INK,
             y=1.02)
plt.tight_layout()

pdf_path = OUT / "training_curves_v12.pdf"
png_path = OUT / "training_curves_v12.png"
fig.savefig(pdf_path, facecolor="white", bbox_inches="tight")
fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")
print(f"Saved {pdf_path}")
print(f"Saved {png_path}")
