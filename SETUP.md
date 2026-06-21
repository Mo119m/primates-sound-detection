# Environment setup

You have already downloaded and installed **Miniconda** from the official site.
Follow these steps once per machine. Run them in **Anaconda Prompt** (Windows)
or **Terminal** (macOS/Linux).

## 1. Create and activate the environment

```bash
conda create -n primates python=3.12 -y
conda activate primates
```

Your prompt should now start with `(primates)`. Keep it active for all the
commands below. (Python 3.12 matches the training environment exactly.)

## 2. Get the code

```bash
git clone https://github.com/mo119m/primates-sound-detection.git
cd primates-sound-detection
```

(Already have the folder? Just `cd` into it and `git pull`.)

## 3. Install the packages

```bash
pip install -r requirements-frozen.txt
pip install jupyter
```

> Use **pip**, not `conda install`, for these packages. The versions are pinned
> in `requirements-frozen.txt` and require **Keras 3 / TensorFlow ≥ 2.16** —
> that is what loads `best_model_v12.h5`.

## 4. Check it worked

```bash
python -c "import tensorflow as tf, keras; print('TF', tf.__version__, '| Keras', keras.__version__)"
```

Expected: `TF 2.20.0 | Keras 3.13.2`. (If Keras shows 2.x, you have the wrong
TensorFlow — reinstall with step 3.)

## 5. Run the notebook

```bash
jupyter notebook
```

This opens Jupyter in your browser. Open
`main_pipeline_notebooks/main_local.ipynb`, set the data paths for this machine
in **cell 2**, then run the cells top to bottom.

---

## Optional: GPU acceleration

The pipeline works on CPU out of the box. If your machine has a supported GPU,
you can speed up training and detection significantly.

### macOS — Apple Silicon (M1 / M2 / M3 / M4)

```bash
conda activate primates
pip install tensorflow-metal
```

That's it. Restart Jupyter and the notebook's GPU-check cell will show your GPU.

### Linux — NVIDIA GPU

```bash
conda activate primates
pip install tensorflow[and-cuda]
```

This replaces the CPU-only TensorFlow with a version that bundles CUDA/cuDNN.

### Windows — NVIDIA GPU

TensorFlow on native Windows has been CPU-only since version 2.11. Two options:

1. **WSL2 (recommended)**: Install [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)
   with Ubuntu, then follow the **Linux** instructions above inside WSL.
2. **CPU is fine**: For loading a pre-trained model and running detection, CPU
   is fast enough. GPU mainly helps during training (`FORCE_RETRAIN = True`).

### No GPU? No problem

Inference (loading model → detecting in audio) runs fine on CPU. A 10-minute
WAV file takes roughly 20–30 seconds on a modern laptop CPU. Training from
scratch is slower (~1–2 hours vs ~15 min on GPU) but is a one-time cost.

---

## Coming back later

After the first-time setup, every session is just:

```bash
conda activate primates
cd primates-sound-detection
jupyter notebook
```
