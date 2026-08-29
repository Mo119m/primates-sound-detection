"""
Build the spectrogram data-augmentation figure for the manuscript (V12).

Restrained, near-monochrome journal style: a single serif typeface, one text
colour (no grey text), thin hairline rules, neutral light fills, and hatching
(rather than colour) to distinguish spectrogram bands. Minimal ornament.

  (A) Standard augmentation -- four operations cycled to a 3,000-row target,
      all target classes.
  (B) High-frequency nuisance augmentation -- +2x, Colobus_guereza only:
      the mel band above 1.5 kHz is replaced with the high band of a random
      background clip, leaving the low-frequency roar intact.

Outputs:
  figures/augmentation_v12.pdf
  figures/augmentation_v12.png
"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

# ── typography ────────────────────────────────────────────────────────────
import _figstyle
_figstyle.apply()

OUT = Path(__file__).parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── palette: one ink colour for all text/rules; neutral grey fills ─────────
INK = "#222222"        # the single text / rule colour
LINE = "#222222"
FRAME = "#8C8C8C"      # thin box borders
FILL = "#F0EFEC"       # neutral low-band / card fill
HATCH = "#9A9A9A"      # hatch lines for the high band

fig, ax = plt.subplots(figsize=(6.50, 4.16), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
fig.patch.set_facecolor("white")

ax.text(50, 97, "Spectrogram data augmentation",
        ha="center", va="center", color=INK)

# ══════════════════════════════════════════════════════════════════════════
# (A) Standard augmentation -- four operations cycled, all target classes
# ══════════════════════════════════════════════════════════════════════════
ax.text(4, 89, "(A)  Standard augmentation",
        ha="left", va="center", color=INK)
ax.text(4, 84.7, "four operations cycled to 3,000 per class, all target classes",
        ha="left", va="center", color=INK, style="normal")

# Bodies are two lines. At the shared 9 pt scale the one-line versions
# overran the card borders, and widening five cards would have cost the
# figure its margins.
# Order and labels follow scripts/pack_v13_images.py, which passes aug 0
# through untouched and then selects the transformation as ``kind = aug % 4``.
# The ×1/×3 multipliers this row used to carry belong to the legacy
# src/augmentation.py path (config.AUGMENTATION_CONFIG), which the v13 packer
# that built the shipped set never calls.
cards = [
    ("Original", "pass\nthrough", "aug 0"),
    ("Time crop", "5–10 % of\ntime axis", "aug 1"),
    ("Frequency crop", "5–10 % of\nmel axis", "aug 2"),
    ("Frequency shift", "±9 mel\nbins", "aug 3"),
    ("Background mix", "SNR −5 to\n10 dB", "aug 4"),
]
n = len(cards)
gap = 1.8
cw = (92 - (n - 1) * gap) / n
cy, ch = 65, 15.5
x0 = 4
for i, (title, body, mult) in enumerate(cards):
    x = x0 + i * (cw + gap)
    ax.add_patch(Rectangle((x, cy), cw, ch, facecolor="white",
                           edgecolor=FRAME, linewidth=0.9, zorder=2))
    ax.text(x + cw / 2, cy + ch - 2.2, title, ha="center", va="top", color=INK)
    ax.text(x + cw / 2, cy + ch / 2 - 1.3, body, ha="center", va="center", color=INK, linespacing=1.3)
    ax.text(x + cw / 2, cy + 1.7, mult, ha="center", va="center", color=INK)

ax.text(50, 60.5, "cycled in this order until the class reaches 3,000 examples",
        ha="center", va="center", color=INK, style="normal")

ax.plot([4, 96], [54, 54], color=FRAME, linewidth=0.6, zorder=1)

# ══════════════════════════════════════════════════════════════════════════
# (B) High-frequency nuisance augmentation -- +2x, Colobus only
# ══════════════════════════════════════════════════════════════════════════
ax.text(4, 50, "(B)  High-frequency nuisance augmentation",
        ha="left", va="center", color=INK)
ax.text(4, 45.4, "+2× per reference clip, Colobus guereza only — in the released code, not in the packed set measured here",
        ha="left", va="center", color=INK, style="normal")

SPEC_Y, SPEC_H = 8, 28
DIV = 0.45                       # 1.5 kHz sits at ~45 % of the mel axis


def draw_spec(x, w, hatch):
    """Schematic mel-spectrogram: plain low band, hatched high band."""
    yb, yt = SPEC_Y, SPEC_Y + SPEC_H
    yd = yb + DIV * SPEC_H
    # low band -- kept (neutral fill, no texture)
    ax.add_patch(Rectangle((x, yb), w, yd - yb, facecolor=FILL,
                           edgecolor="none", zorder=2))
    # high band -- hatched (pattern encodes its content)
    ax.add_patch(Rectangle((x, yd), w, yt - yd, facecolor="white",
                           edgecolor=HATCH, hatch=hatch, linewidth=0.0,
                           zorder=2))
    # outer frame + 1.5 kHz divider
    ax.add_patch(Rectangle((x, yb), w, SPEC_H, fill=False, edgecolor=FRAME,
                           linewidth=0.9, zorder=4))
    ax.plot([x, x + w], [yd, yd], color=INK, linestyle=(0, (4, 2.5)),
            linewidth=0.9, zorder=5)
    return yb, yt, yd


ox, ow = 8, 30
yb, yt, yd = draw_spec(ox, ow, hatch="....")
ax.text(ox + ow / 2, yt + 2.2, "Colobus reference clip", ha="center",
        va="bottom", color=INK)
ax.text(ox - 1.2, yb, "20 Hz", ha="right", va="center", color=INK)
ax.text(ox - 1.2, yd, "1.5 kHz", ha="right", va="center", color=INK)
ax.text(ox - 1.2, yt, "8 kHz", ha="right", va="center", color=INK)

ax.add_patch(FancyArrowPatch((ox + ow + 2, yb + SPEC_H / 2),
                             (ox + ow + 12, yb + SPEC_H / 2),
                             arrowstyle="-|>,head_length=6,head_width=3.2",
                             color=INK, linewidth=1.1, zorder=6))
ax.text(ox + ow + 7, yb + SPEC_H / 2 + 2.2, "replace\nhigh band", ha="center",
        va="bottom", color=INK, style="normal", linespacing=1.1)

ax2, aw = ox + ow + 14, 30
yb2, yt2, yd2 = draw_spec(ax2, aw, hatch="////")
ax.text(ax2 + aw / 2, yt2 + 2.2, "Augmented variant", ha="center",
        va="bottom", color=INK)

ax.text(ax2 + aw + 2, yd2 + 0.5 * (yt2 - yd2),
        "> 1.5 kHz replaced\nwith random background", ha="left", va="center", color=INK, linespacing=1.2)
ax.text(ax2 + aw + 2, yb2 + 0.5 * (yd2 - yb2),
        "low band kept\n(low-frequency roar)", ha="left", va="center", color=INK, linespacing=1.2)

ax.text(50, 2.6,
        "Decorrelates high-frequency texture from the Colobus "
        "label; not applied to Cernic, whose calls occupy higher frequencies.",
        ha="center", va="center", color=INK)

plt.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.01)

pdf_path = OUT / "augmentation_v12.pdf"
png_path = OUT / "augmentation_v12.png"
fig.savefig(pdf_path, facecolor="white", bbox_inches="tight")
fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")
print(f"Saved {pdf_path}")
print(f"Saved {png_path}")
