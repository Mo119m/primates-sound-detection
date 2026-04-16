"""
Annotate the 4-panel spectrogram figure with arrows pointing out
the distinctive feature of each species.

Run this AFTER the cell that generates 03_example_spectrograms.png
in 01_data_overview.ipynb, or provide any 4-panel spectrogram image.

Usage in Colab:
  1. Run the spectrogram cell first (produces the figure or axes)
  2. Paste this code into the next cell
  3. It loads the saved PNG and overlays annotations
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyArrowPatch

# --- Load the base image -------------------------------------------------
img = mpimg.imread('figures/03_example_spectrograms.png')
h, w = img.shape[:2]

fig, ax = plt.subplots(figsize=(10, 14), dpi=180)
ax.imshow(img)
ax.axis('off')

# --- Annotation config ----------------------------------------------------
FONT = dict(fontsize=11, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.75))

ARROW = dict(arrowstyle='->', color='white', linewidth=2.0,
             connectionstyle='arc3,rad=0.15')

# Coordinates are in pixel space of the image.
# Each entry: (arrow_start_xy, arrow_end_xy, label, label_xy)
# Adjust these values after seeing the actual image dimensions.
#
# The image is ~800px wide, ~1100px tall, with 4 stacked spectrograms.
# Each panel is roughly 250px tall. Spectrograms occupy x ≈ 50..680.

annotations = [
    {
        'label': 'Short alarm call\n(pyow ~0.1 s)',
        'label_xy': (w * 0.82, h * 0.10),
        'arrow_to': (w * 0.37, h * 0.08),   # the bright burst in Cerco
    },
    {
        'label': 'Sustained harmonic\nbands (1–4 kHz)',
        'label_xy': (w * 0.82, h * 0.34),
        'arrow_to': (w * 0.50, h * 0.33),   # mid-freq bands in Colobus
    },
    {
        'label': 'Dense pant-hoot\nsequence',
        'label_xy': (w * 0.82, h * 0.56),
        'arrow_to': (w * 0.35, h * 0.56),   # repeated vertical bars in Pan
    },
    {
        'label': 'No primate call\n(insects / ambient)',
        'label_xy': (w * 0.82, h * 0.82),
        'arrow_to': (w * 0.50, h * 0.82),   # flat background
    },
]

for ann in annotations:
    ax.annotate(
        ann['label'],
        xy=ann['arrow_to'],
        xytext=ann['label_xy'],
        fontsize=11, fontweight='bold', color='white',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='black', alpha=0.78),
        arrowprops=dict(arrowstyle='->', color='white', lw=2.2,
                        connectionstyle='arc3,rad=-0.15'),
        ha='left', va='center',
    )

plt.tight_layout()
out = 'figures/03_example_spectrograms_annotated.png'
fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
plt.show()
print(f'Saved {out}')
