"""
Build the spectrogram data-augmentation figure for the manuscript (V12).

Restrained, near-monochrome journal style: a single serif typeface, one text
colour (no grey text), thin hairline rules, neutral light fills, and hatching
(rather than colour) to distinguish spectrogram bands. Minimal ornament.

  (A) Standard augmentation -- 7x per reference clip, all target classes.
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
_SERIF_PREF = ["Times New Roman", "Liberation Serif", "Nimbus Roman",
               "STIXGeneral", "DejaVu Serif"]
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = _SERIF_PREF
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["pdf.fonttype"] = 42  # embed TrueType (avoid Type 3; Elsevier requirement)
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["hatch.linewidth"] = 0.5

OUT = Path(__file__).parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── palette: one ink colour for all text/rules; neutral grey fills ─────────
INK = "#222222"        # the single text / rule colour
LINE = "#222222"
FRAME = "#8C8C8C"      # thin box borders
FILL = "#F0EFEC"       # neutral low-band / card fill
HATCH = "#9A9A9A"      # hatch lines for the high band

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
ax.text(4, 84.7, "7× per reference clip, all target classes",
        ha="left", va="center", fontsize=9, color=INK, style="italic")

cards = [
    ("Original", "pass through", "×1"),
    ("Background mix", "SNR −5 to 10 dB", "×3"),
    ("Time crop", "10–30 % of time axis", "×1"),
    ("Frequency crop", "10–30 % of mel axis", "×1"),
    ("Frequency shift", "±20 mel bins", "×1"),
]
n = len(cards)
gap = 1.8
cw = (92 - (n - 1) * gap) / n
cy, ch = 66, 13.5
x0 = 4
for i, (title, body, mult) in enumerate(cards):
    x = x0 + i * (cw + gap)
    ax.add_patch(Rectangle((x, cy), cw, ch, facecolor="white",
                           edgecolor=FRAME, linewidth=0.9, zorder=2))
    ax.text(x + cw / 2, cy + ch - 2.2, title, ha="center", va="top",
            fontsize=9.2, color=INK)
    ax.text(x + cw / 2, cy + ch / 2 - 1.3, body, ha="center", va="center",
            fontsize=7.6, color=INK, linespacing=1.3)
    ax.text(x + cw / 2, cy + 1.7, mult, ha="center", va="center",
            fontsize=9.2, color=INK)

ax.annotate("", xy=(97.6, cy + ch / 2), xytext=(96.4, cy + ch / 2),
            arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=1.1))
ax.text(50, 60.5, "= 7 variants per reference clip",
        ha="center", va="center", fontsize=9, color=INK, style="italic")

ax.plot([4, 96], [54, 54], color=FRAME, linewidth=0.6, zorder=1)

# ══════════════════════════════════════════════════════════════════════════
# (B) High-frequency nuisance augmentation -- +2x, Colobus only
# ══════════════════════════════════════════════════════════════════════════
ax.text(4, 49, "(B)  High-frequency nuisance augmentation",
        ha="left", va="center", fontsize=11.5, color=INK)
ax.text(4, 44.7, "+2× per reference clip, Colobus guereza only",
        ha="left", va="center", fontsize=9, color=INK, style="italic")

SPEC_Y, SPEC_H = 8, 30
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
        va="bottom", fontsize=8.5, color=INK)
ax.text(ox - 1.2, yb, "20 Hz", ha="right", va="center", fontsize=7, color=INK)
ax.text(ox - 1.2, yd, "1.5 kHz", ha="right", va="center", fontsize=7, color=INK)
ax.text(ox - 1.2, yt, "8 kHz", ha="right", va="center", fontsize=7, color=INK)

ax.add_patch(FancyArrowPatch((ox + ow + 2, yb + SPEC_H / 2),
                             (ox + ow + 12, yb + SPEC_H / 2),
                             arrowstyle="-|>,head_length=6,head_width=3.2",
                             color=INK, linewidth=1.1, zorder=6))
ax.text(ox + ow + 7, yb + SPEC_H / 2 + 2.2, "replace\nhigh band", ha="center",
        va="bottom", fontsize=7.2, color=INK, style="italic", linespacing=1.1)

ax2, aw = ox + ow + 14, 30
yb2, yt2, yd2 = draw_spec(ax2, aw, hatch="////")
ax.text(ax2 + aw / 2, yt2 + 2.2, "Augmented variant", ha="center",
        va="bottom", fontsize=8.5, color=INK)

ax.text(ax2 + aw + 2, yd2 + 0.5 * (yt2 - yd2),
        "> 1.5 kHz replaced\nwith random background", ha="left", va="center",
        fontsize=7.6, color=INK, linespacing=1.2)
ax.text(ax2 + aw + 2, yb2 + 0.5 * (yd2 - yb2),
        "low band kept\n(low-frequency roar)", ha="left", va="center",
        fontsize=7.6, color=INK, linespacing=1.2)

ax.text(50, 2.6,
        "Decorrelates high-frequency texture from the Colobus "
        "label; not applied to Cernic, whose calls occupy higher frequencies.",
        ha="center", va="center", fontsize=7.8, color=INK)

plt.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.01)

pdf_path = OUT / "augmentation_v12.pdf"
png_path = OUT / "augmentation_v12.png"
fig.savefig(pdf_path, facecolor="white", bbox_inches="tight")
fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")
print(f"Saved {pdf_path}")
print(f"Saved {png_path}")
