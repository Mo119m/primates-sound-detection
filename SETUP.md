# Environment setup

You have already downloaded and installed **Miniconda** from the official site.
Follow these steps once per machine. Run them in **Anaconda Prompt** (Windows)
or **Terminal** (macOS/Linux).

## 1. Create and activate the environment

```bash
conda create -n primates python=3.10 -y
conda activate primates
```

Your prompt should now start with `(primates)`. Keep it active for all the
commands below.

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

Expected: `TF 2.17.0 | Keras 3.x`. (If Keras shows 2.x, you have the wrong
TensorFlow — reinstall with step 3.)

## 5. Run the notebook

```bash
jupyter notebook
```

This opens Jupyter in your browser. Open
`main_pipeline_notebooks/main_local.ipynb`, set the data paths for this machine
in **cell 2**, then run the cells top to bottom.

---

## Coming back later

After the first-time setup, every session is just:

```bash
conda activate primates
cd primates-sound-detection
jupyter notebook
```
