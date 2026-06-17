"""
Build the spectrogram data-augmentation figure for the manuscript (V12).

Two parts, in the restrained academic style shared with the confusion-matrix
and training-curve figures (Liberation Serif, muted palette, hairline frames):

  (A) Standard augmentation -- 7x per reference clip, applied to all target
      classes (Cernic, Colobus_guereza, Colobus_confuser).
  (B) High-frequency nuisance augmentation -- +2x per reference clip, applied
      only to Colobus_guereza: the mel band above 1.5 kHz is replaced with the
      high band of a random background clip, leaving the low-frequency roar
      intact. This decorrelates high-frequency texture from the Colobus label.

Outputs:
  figures/augmentation_v12.pdf
  figures/augmentation_v12.png
"""
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

# ── typography ────────────────────────────────────────────────────────────
_SERIF_PREF = ["Times New Roman", "Liberation Serif", "Nimbus Roman",
               "STIXGeneral", "DejaVu Serif"]
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = _SERIF_PREF
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── palette ───────────────────────────────────────────────────────────────
INK = "#1E2A32"
SUBINK = "#5A6670"
BODY = "#33373A"
PAPER = "#FBFAF7"
FRAME = "#CBC7BF"
ACCENT = "#3E5C76"      # standard-aug accent
LOW = "#5B7553"         # kept low-frequency roar (muted green)
HIGH = "#99584B"        # replaced high band (muted brick)

fig, ax = plt.subplots(figsize=(10, 6.4), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
fig.patch.set_facecolor("white")

ax.text(50, 97, "Spectrogram data augmentation",
        ha="center", va="center", fontsize=15, color=INK)

# ══════════════════════════════════════════════════════════════════════════
# (A) Standard augmentation -- 7x, all target classes
# ══════════════════════════════════════════════════════════════════════════
ax.text(4, 89, "(A)  Standard augmentation",
        ha="left", va="center", fontsize=11.5, color=INK)
ax.text(4, 84.7,
        "7$\\times$ per reference clip · applied to all target classes",
        ha="left", va="center", fontsize=9, color=SUBINK, style="italic")

cards = [
    ("Original", "pass through", "$\\times$1"),
    ("Background mix", "SNR $-5$ to 10 dB", "$\\times$3"),
    ("Time crop", "10–30 % of time axis", "$\\times$1"),
    ("Frequency crop", "10–30 % of mel axis", "$\\times$1"),
    ("Frequency shift", "$\\pm$20 mel bins", "$\\times$1"),
]
n = len(cards)
gap = 1.8
cw = (92 - (n - 1) * gap) / n
cy, ch = 66, 13.5
x0 = 4
for i, (title, body, mult) in enumerate(cards):
    x = x0 + i * (cw + gap)
    ax.add_patch(FancyBboxPatch(
        (x, cy), cw, ch, boxstyle="round,pad=0.25,rounding_size=0.4",
        facecolor=PAPER, edgecolor=FRAME, linewidth=1.0, zorder=2))
    # thin accent rule along the top
    ax.plot([x + 0.6, x + cw - 0.6], [cy + ch - 0.5, cy + ch - 0.5],
            color=ACCENT, linewidth=2.2, zorder=3, solid_capstyle="round")
    ax.text(x + cw / 2, cy + ch - 2.4, title, ha="center", va="top",
            fontsize=9.2, color=INK)
    ax.text(x + cw / 2, cy + ch / 2 - 1.4, body, ha="center", va="center",
            fontsize=7.6, color=BODY, linespacing=1.3)
    ax.text(x + cw / 2, cy + 1.6, mult, ha="center", va="center",
            fontsize=9.5, color=ACCENT)

ax.annotate("", xy=(97.5, cy + ch / 2), xytext=(96.2, cy + ch / 2),
            arrowprops=dict(arrowstyle="-|>", color=SUBINK, linewidth=1.2))
ax.text(50, 60.5, "= 7 variants per reference clip",
        ha="center", va="center", fontsize=9, color=SUBINK, style="italic")

# divider rule between the two parts
ax.plot([4, 96], [54, 54], color=FRAME, linewidth=0.8, zorder=1)

# ══════════════════════════════════════════════════════════════════════════
# (B) High-frequency nuisance augmentation -- +2x, Colobus only
# ══════════════════════════════════════════════════════════════════════════
ax.text(4, 49, "(B)  High-frequency nuisance augmentation",
        ha="left", va="center", fontsize=11.5, color=INK)
ax.text(4, 44.7,
        "+2$\\times$ per reference clip · Colobus guereza only",
        ha="left", va="center", fontsize=9, color=SUBINK, style="italic")

# mel-spectrogram schematic; on a mel axis 1.5 kHz sits at ~45 % of the height
SPEC_Y, SPEC_H = 8, 30
DIV = 0.45                      # fraction of height for the 1.5 kHz divider
rng = np.random.default_rng(7)


def draw_spec(x, w, replaced):
    """Draw a schematic mel-spectrogram with a low (kept) and high band."""
    yb, yt = SPEC_Y, SPEC_Y + SPEC_H
    yd = yb + DIV * SPEC_H
    # low band -- kept low-frequency roar
    ax.add_patch(Rectangle((x, yb), w, yd - yb, facecolor=LOW, alpha=0.16,
                           edgecolor="none", zorder=2))
    # high band -- bird/insect texture or replacement
    ax.add_patch(Rectangle((x, yd), w, yt - yd, facecolor=HIGH, alpha=0.14,
                           edgecolor="none", zorder=2))
    # harmonic roar bands in the low region (identical in both panels)
    for fr in (0.18, 0.30, 0.42):
        yy = yb + fr * (yd - yb)
        ax.plot([x + 0.04 * w, x + 0.96 * w], [yy, yy], color=LOW,
                linewidth=1.4, alpha=0.7, zorder=3, solid_capstyle="round")
    # high-band "noise" marks; a different random pattern when replaced
    seed = 23 if replaced else 11
    r = np.random.default_rng(seed)
    xs = x + (0.06 + 0.88 * r.random(70)) * w
    ys = yd + (0.06 + 0.88 * r.random(70)) * (yt - yd)
    ax.scatter(xs, ys, s=2.0, color=HIGH, alpha=0.55, zorder=3, linewidths=0)
    # frame + divider
    ax.add_patch(Rectangle((x, yb), w, SPEC_H, fill=False, edgecolor=FRAME,
                           linewidth=1.0, zorder=4))
    ax.plot([x, x + w], [yd, yd], color=INK, linestyle=(0, (4, 2.5)),
            linewidth=1.0, zorder=5)
    return yb, yt, yd


# original Colobus clip
ox, ow = 8, 30
yb, yt, yd = draw_spec(ox, ow, replaced=False)
ax.text(ox + ow / 2, yt + 2.2, "Colobus reference clip", ha="center",
        va="bottom", fontsize=8.5, color=INK)
# frequency labels (left of the original)
ax.text(ox - 1.2, yb, "20 Hz", ha="right", va="center", fontsize=7, color=SUBINK)
ax.text(ox - 1.2, yd, "1.5 kHz", ha="right", va="center", fontsize=7, color=INK)
ax.text(ox - 1.2, yt, "8 kHz", ha="right", va="center", fontsize=7, color=SUBINK)

# arrow to the augmented clip
ax.add_patch(FancyArrowPatch((ox + ow + 2, yb + SPEC_H / 2),
                             (ox + ow + 12, yb + SPEC_H / 2),
                             arrowstyle="-|>,head_length=6,head_width=3.4",
                             color=SUBINK, linewidth=1.3, zorder=6))
ax.text(ox + ow + 7, yb + SPEC_H / 2 + 2.4, "replace\nhigh band", ha="center",
        va="bottom", fontsize=7.2, color=SUBINK, style="italic", linespacing=1.1)

# augmented clip
ax2, aw = ox + ow + 14, 30
yb2, yt2, yd2 = draw_spec(ax2, aw, replaced=True)
ax.text(ax2 + aw / 2, yt2 + 2.2, "Augmented variant", ha="center",
        va="bottom", fontsize=8.5, color=INK)

# band annotations (to the right of the augmented clip)
ax.text(ax2 + aw + 2, yb2 + 0.5 * (yd2 - yb2),
        "low band kept\n(low-frequency roar)", ha="left", va="center",
        fontsize=7.6, color=LOW, linespacing=1.2)
ax.text(ax2 + aw + 2, yd2 + 0.5 * (yt2 - yd2),
        "> 1.5 kHz replaced with\nrandom background", ha="left", va="center",
        fontsize=7.6, color=HIGH, linespacing=1.2)

ax.text(50, 3.0,
        "Decorrelates high-frequency texture from the $\\mathit{Colobus}$ label; "
        "not applied to Cernic, whose calls occupy higher frequencies.",
        ha="center", va="center", fontsize=7.8, color=SUBINK)

plt.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.01)

pdf_path = OUT / "augmentation_v12.pdf"
png_path = OUT / "augmentation_v12.png"
fig.savefig(pdf_path, facecolor="white", bbox_inches="tight")
fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")
print(f"Saved {pdf_path}")
print(f"Saved {png_path}")
