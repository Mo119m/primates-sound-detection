"""
One type scale for every manuscript figure.

The earlier approach -- multiplying every fontsize literal by a constant -- was
wrong twice over. It widened the gaps in an already inconsistent hierarchy, and
it left the figures physically oversized, so matplotlib's own layout squeezed
the plotting area to make room for the larger text (Figure 3 lost most of its
x-ticks that way).

The fix is to stop scaling. A figure drawn at the width it will occupy on the
page is reproduced at 1:1, so a size set here in points is that many points in
the PDF. Elsevier's single-column text block in `elsarticle` with the `3p`
option is about 6.5 in wide, and captions set at roughly 9 pt, so the scale
below puts axis text at 9-10 pt: legible, and matched to the caption rather than
shrunk below it, which is what the co-author review asked for.

Import and call `apply()` before creating any figure. Use `TEXT_WIDTH_IN` for
the figure width and never pass an explicit `fontsize=` afterwards; the point of
a single scale is that it is not overridden locally.
"""
import matplotlib as mpl

TEXT_WIDTH_IN = 6.5          # elsarticle 3p single-column text block

_SERIF = ["Times New Roman", "Liberation Serif", "Nimbus Roman",
          "STIXGeneral", "DejaVu Serif"]

# One step between levels, no more. A figure with five different sizes in it
# reads as decorated rather than typeset.
BASE = 9.0        # annotations, tick labels
LABEL = 10.0      # axis labels, in-figure body text
HEAD = 11.0       # panel headings
TITLE = 12.0      # figure title


def apply():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": _SERIF,
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        # Elsevier rejects Type 3; 42 embeds TrueType subsets.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": BASE,
        "axes.titlesize": HEAD,
        "axes.labelsize": LABEL,
        "xtick.labelsize": BASE,
        "ytick.labelsize": BASE,
        "legend.fontsize": BASE,
        "figure.titlesize": TITLE,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


# Shared palette, so the figures look like a set rather than three drawings.
TRAIN = "#3E5C76"
VAL = "#99584B"
INK = "#1E2A32"
SUBINK = "#5A6670"
FRAME = "#CBC7BF"
GRID = "#ECEAE4"
