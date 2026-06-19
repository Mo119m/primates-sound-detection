# Environment setup (detection machines)

This is the turnkey recipe for installing the detection environment on a fresh
machine (Anaconda/Miniconda). It avoids the version conflicts that come from
letting conda resolve the scientific stack.

## The one rule

**Use conda only to create a clean Python interpreter. Install everything else
with `pip` from `requirements-frozen.txt`. Never run `conda install
tensorflow`.**

Why: the conda channels ship an old, Keras-2 build of TensorFlow (2.15 or
older), which **cannot load `best_model_v12.h5`** (it is serialised by Keras 3).
Mixing `conda install` and `pip install` for numpy/protobuf/TensorFlow also
causes ABI clashes (`numpy.dtype size changed`, protobuf errors). Keeping conda
to just the interpreter and pip for the rest eliminates both problems.

## Steps (every machine)

```bash
# 1. Create a clean environment with Python only (no ML packages from conda)
conda create -n primates python=3.10 -y
conda activate primates

# 2. Install the exact pinned stack with pip
pip install -r requirements-frozen.txt

# 3. Verify the versions and that the model loads
python -c "import tensorflow as tf, keras; print('TF', tf.__version__, '| Keras', keras.__version__)"
```

Expected: TensorFlow 2.17.x and Keras 3.x. If you see Keras 2.x, you installed
the wrong TensorFlow (probably via conda) — remove it and reinstall with pip.

## Confirm the version matches the training machine

The pinned TensorFlow version must produce a Keras version that can read the
checkpoint. On the **training machine** (Colab), run:

```python
import tensorflow as tf, keras, numpy as np
print("TF:", tf.__version__, "| Keras:", keras.__version__, "| NumPy:", np.__version__)
```

If it reports a TensorFlow version other than 2.17.0, edit the `tensorflow==`
line in `requirements-frozen.txt` to match, then reinstall on the detection
machines.

## Quick model-load smoke test

After install, point this at the model file to confirm it loads end to end:

```bash
python -c "from src.model import load_model; load_model('PATH/TO/best_model_v12.h5')"
```

A successful load prints "Model loaded successfully!" and confirms the Keras
version is compatible.

## Multi-machine note

Each detection machine reads data from its own hard drive. The data paths are
set per machine in `main_pipeline_notebooks/main_local.ipynb` (cell 2 has a
comment/uncomment template). The Python environment created above is identical
on every machine; only the data paths differ.
