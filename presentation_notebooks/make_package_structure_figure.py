"""
Build a backup slide describing the package structure.

Produces presentation_notebooks/figures/package_structure.png.
Layout: centered title, one row per module with the file name on the
left (monospace) and a one-line plain-English description on the right.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUT = Path(__file__).parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)


MODULES = [
    ("config.py",        "All paths, hyper-parameters, and species/background folder list — one place to change for a new dataset."),
    ("data_loader.py",   "Recursively scans species + background folders, resamples every clip to 44.1 kHz, returns NumPy arrays."),
    ("preprocessing.py", "Audio → mel-spectrogram → 224×224 RGB image; also generates sliding windows for detection."),
    ("augmentation.py",  "Background mixing, time/frequency cropping, frequency translation; ×7 effective training-set multiplier."),
    ("model.py",         "VGG19 base + custom head, compile step, callbacks, and a helper to unfreeze the last block for fine-tuning."),
    ("train.py",         "End-to-end training pipeline: load → augment → split → train → evaluate → save best model + history."),
    ("detection.py",     "Sliding-window inference on long audio, threshold sweep, per-species non-maximum suppression, CSV export."),
    ("utils.py",         "Result visualisation, clip extraction for manual review, detection-statistics reporting."),
]

fig, ax = plt.subplots(figsize=(12, 6.5), dpi=200)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# Title
ax.text(50, 94,
        "Reusable Python package  (src/)",
        ha="center", va="center", fontsize=17, fontweight="bold",
        color="#2C3E50")
ax.text(50, 87,
        "One module per stage of the pipeline — drop in your own primate recordings by editing config.py",
        ha="center", va="center", fontsize=11, color="#7F8C8D", style="italic")

# Table rows
row_top = 80
row_h = 8.5
name_col_x = 5
name_col_w = 22
desc_col_x = 30

for i, (name, desc) in enumerate(MODULES):
    y_top = row_top - i * row_h
    y_center = y_top - row_h / 2

    # Row background (alternating)
    if i % 2 == 0:
        ax.add_patch(FancyBboxPatch(
            (name_col_x - 2, y_top - row_h + 1), 94, row_h - 1.5,
            boxstyle="round,pad=0.1,rounding_size=0.4",
            facecolor="#F8F9FA", edgecolor="none", zorder=0,
        ))

    # Module name (monospace, colored)
    ax.text(name_col_x, y_center, name,
            ha="left", va="center",
            fontsize=12, fontweight="bold",
            family="monospace", color="#2980B9")

    # Description
    ax.text(desc_col_x, y_center, desc,
            ha="left", va="center",
            fontsize=10.5, color="#2C3E50")

# Footer
ax.text(50, 4,
        "Total: ~1200 lines of Python  |  MIT-style licence  |  one-command Colab reproduction",
        ha="center", va="center", fontsize=10,
        color="#95A5A6", style="italic")


plt.tight_layout()
out_path = OUT / "package_structure.png"
fig.savefig(out_path, bbox_inches="tight", dpi=220, facecolor="white")
print(f"Saved {out_path}")
