"""
Generate colab/v13_train.ipynb.

The notebook is generated rather than hand-edited because a notebook is JSON:
editing it by hand invites broken cells and unreviewable diffs, and the shell
commands in it have to stay in step with the scripts in ``scripts/``. Run this
after changing either.

    python colab/make_notebook.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip().split("\n")}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip().split("\n")}


CELLS = [
    md("""
# V13 — train and score leave-one-station-out

This runs on Colab because of two hard limits on the laptop the data lives on:
VGG19 backpropagates at **2.7 images/s** there, so a sixteen-fold sweep is about
**113 hours**, and the head alone still needs **6 hours per fold**.

## What to upload first

Produce these two locally (about 20 minutes), then put them in Drive under
`MyDrive/primates_v13/`:

```bash
python scripts/build_v13_dataset.py     # -> data/outputs/v13_manifest.csv
python scripts/pack_v13_images.py       # -> data/outputs/v13_images.npy  (4.77 GB)
                                        #    data/outputs/v13_index.csv
```

Only those go up. The 444 GB of field recordings stay on the external drive —
nothing here needs them, because every clip has already been reduced to the exact
2 s analysis window the model sees, verified sample-for-sample against the source
recordings.

## What comes back

`v13_loso.csv`: per held-out station, how many of that station's reviewed false
positives the new model rejects and how many of its confirmed calls it keeps.
That is **precision**, measured out of sample. It is not recall — a call V12
never fired on was never exported and never reviewed, so no experiment on this
set can see it.
"""),

    code("""
!nvidia-smi -L
import tensorflow as tf
print("TF", tf.__version__, "GPUs:", tf.config.list_physical_devices('GPU'))
"""),

    md("## 1. Repository and data"),

    code("""
from google.colab import drive
drive.mount('/content/drive')

DRIVE = '/content/drive/MyDrive/primates_v13'
!ls -la $DRIVE
"""),

    code("""
# The repo. Use your branch if the V13 scripts are not on main yet.
!git clone https://github.com/mo119m/primates-sound-detection.git /content/repo 2>/dev/null || true
%cd /content/repo
!git pull --ff-only 2>/dev/null || true
!pip -q install librosa soundfile
"""),

    code("""
import os, shutil
os.makedirs('/content/repo/data/outputs', exist_ok=True)

# Copy off Drive to local disk first: training reads the feature cache every
# epoch, and Drive's FUSE mount makes that far slower than the GPU.
for name in ('v13_images.npy', 'v13_index.csv'):
    src, dst = f'{DRIVE}/{name}', f'/content/repo/data/outputs/{name}'
    if not os.path.exists(dst):
        print('copying', name); shutil.copy(src, dst)
print(os.popen('ls -la /content/repo/data/outputs').read())
"""),

    md("""
## 2. Feature cache

The VGG19 base is frozen for stage 1, so its output for a given image is the same
in every epoch and every fold. Computing it once and keeping the `block4_conv4`
activations is not an approximation of stage 1 — it *is* stage 1, with the
constant part evaluated once instead of sixteen times over.

About 25 GB at float16. Colab's local disk holds it; Drive should not be used for
this file.
"""),

    code("""
# One fold first, to see a number before committing to the sweep.
!python scripts/train_v13_loso.py --folds IPA20ST --epochs 15 --verbose 1
"""),

    md("""
## 3. The full sweep

Sixteen folds. Each withholds one station from training entirely — including the
1 348 clips whose filenames narrow their origin to a group of stations without
naming one, because the five stations that recorded with GPS off write identical
filenames and guessing between them would leak quietly into every fold.
"""),

    code("""
!python scripts/train_v13_loso.py --folds all --epochs 15
"""),

    code("""
import pandas as pd
df = pd.read_csv('/content/repo/data/outputs/v13_loso.csv')
df
"""),

    code("""
c, f = df['calls'].sum(), df['false_positives'].sum()
kc, kf = df['kept_calls'].sum(), df['kept_false_positives'].sum()
print(f"V12 precision {c/(c+f):.3f}   ({c} calls / {c+f} detections)")
print(f"V13 precision {kc/(kc+kf):.3f}   ({kc} calls / {kc+kf} detections)")
print(f"calls retained {kc/c:.3f}   false positives removed {1-kf/f:.3f}")
print(f"review reduced by {1-(kc+kf)/(c+f):.3f}")
"""),

    md("""
### Excluding IPA4ST

One station holds 2 370 of the 3 654 reviewed false positives — an untrained
species called there in long bouts. It dominates any pooled figure, and the
paper already reports the fifteen-station numbers alongside the sixteen, so both
belong here too.
"""),

    code("""
sub = df[df.station != 'IPA4ST']
c, f = sub['calls'].sum(), sub['false_positives'].sum()
kc, kf = sub['kept_calls'].sum(), sub['kept_false_positives'].sum()
print(f"15 stations, excluding IPA4ST")
print(f"V12 precision {c/(c+f):.3f}  ->  V13 precision {kc/(kc+kf):.3f}")
print(f"calls retained {kc/c:.3f}   false positives removed {1-kf/f:.3f}")
"""),

    md("""
## 4. Stage 2 — unfreezing the last VGG blocks

Everything above trains the head on cached features, which is exactly stage 1.
Stage 2 fine-tunes blocks 4 and 5 at a low learning rate, and cannot use the
cache because those weights move. It needs the images, and it needs this GPU.

Run it only after stage 1 shows a gain worth extending, and read the fold numbers
before deciding: fine-tuning more parameters on the same labels can just as
easily fit the stations it was given.
"""),

    code("""
# Placeholder for the stage-2 sweep. Fill in once stage-1 folds are in:
# the schedule is the V12 one -- unfreeze block4/block5, Adam at 1e-5 --
# applied per fold with the same withheld set.
print('stage 1 first')
"""),

    md("## 5. Save results back to Drive"),

    code("""
!cp /content/repo/data/outputs/v13_loso.csv $DRIVE/
print('saved')
"""),
]

NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

if __name__ == "__main__":
    out = os.path.join(HERE, "v13_train.ipynb")
    with open(out, "w") as fh:
        json.dump(NOTEBOOK, fh, indent=1)
    print(f"wrote {out} ({len(CELLS)} cells)")
