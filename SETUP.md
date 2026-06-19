# Environment setup — complete, foolproof guide

Follow this from top to bottom on **any** machine (Windows, macOS, or Linux;
new or old; with or without a GPU). If you do exactly what is written here, you
will end up with an environment that runs the code and loads the model — no
guesswork about versions.

> **The single most important rule:** install the scientific packages with
> **`pip`**, not with `conda`. Use conda **only** to create a clean Python
> interpreter. `conda install tensorflow` ships an old Keras-2 build that
> **cannot load our model** and drags in conflicting numpy/protobuf. This is the
> #1 cause of "version chaos" on a fresh Anaconda install.

---

## 0. What you are about to install (and why these versions)

| Package | Pinned version | Why it matters |
|---|---|---|
| Python | 3.10 | The version the model was trained/tested on. 3.9–3.12 also work. |
| TensorFlow | 2.17.0 | **Keras 3.** Our model `best_model_v12.h5` is saved by Keras 3 and **must** be loaded by Keras 3 (TF ≥ 2.16). TF ≤ 2.15 is Keras 2 and will fail. |
| numpy | 1.26.4 | Must be < 2.0 to stay ABI-compatible with this TF build. |
| librosa, soundfile, resampy | pinned | Audio loading / resampling. |
| scikit-learn, pandas, matplotlib, opencv | pinned | Feature pipeline, I/O, plots. |
| tensorflow-hub | 0.16.1 | YAMNet filter in the auto-cleanup step. |

All exact versions live in `requirements-frozen.txt`. You never type them by
hand — `pip install -r requirements-frozen.txt` reads them for you.

**About "low version vs high version":** it does not matter what version of
**conda, Anaconda, or Miniconda** you already have — old or new, all work. What
must be exact is the **Python environment we create below** and the **pip
packages**. The recipe pins those, so the result is identical on every machine.

---

## 1. Do you already have conda?

Open a terminal and run:

```bash
conda --version
```

- **It prints a version** (any version — old or new is fine) → skip to step 3.
- **"command not found"** → do step 2 first.

> Terminal = **Anaconda Prompt** on Windows (Start menu → "Anaconda Prompt"),
> or **Terminal** on macOS/Linux.

---

## 2. Install Miniconda (only if you have no conda)

Miniconda is a tiny conda. Download the installer for your OS from
<https://docs.conda.io/en/latest/miniconda.html> and install with all default
options.

- **Windows:** run the `.exe`, click through with defaults, then open
  **"Anaconda Prompt"** from the Start menu.
- **macOS / Linux:** run the downloaded `.sh` script, e.g.
  `bash ~/Downloads/Miniconda3-latest-*.sh`, accept defaults, then **close and
  reopen** the terminal.

Confirm it worked:

```bash
conda --version
```

---

## 3. Create a clean, isolated environment

This makes a fresh Python that cannot be polluted by anything else on the
machine. **Run these two lines:**

```bash
conda create -n primates python=3.10 -y
conda activate primates
```

After `activate`, your prompt should show `(primates)` at the start of the line.
**Every command from now on must be run with `(primates)` showing.**

> Made a mistake and want a clean slate? Delete and start over:
> `conda deactivate && conda env remove -n primates -y`, then redo step 3.

---

## 4. Get the code

```bash
git clone https://github.com/mo119m/primates-sound-detection.git
cd primates-sound-detection
```

(If you already have the folder, just `cd` into it and `git pull`.)

---

## 5. Install everything with pip

```bash
python -m pip install --upgrade pip
pip install -r requirements-frozen.txt
```

This installs the exact pinned stack from step 0. Wait for it to finish (it may
take a few minutes — TensorFlow is large).

> **Do not** run `conda install ...` for any of these packages. If a teammate
> already did and things are broken, the cleanest fix is to delete the env
> (end of step 3) and redo steps 3 and 5.

---

## 6. Verify — three checks, all must pass

**6a. Versions are correct:**

```bash
python -c "import tensorflow as tf, keras, numpy as np; print('TF', tf.__version__, '| Keras', keras.__version__, '| NumPy', np.__version__)"
```

Expected: `TF 2.17.0 | Keras 3.x | NumPy 1.26.4`.
If Keras shows **2.x**, you have the wrong TensorFlow — see Troubleshooting.

**6b. The audio + science stack imports:**

```bash
python -c "import librosa, soundfile, sklearn, pandas, cv2, tensorflow_hub; print('all imports OK')"
```

**6c. The model actually loads** (point the path at your `best_model_v12.h5`):

```bash
python -c "from src.model import load_model; load_model('PATH/TO/best_model_v12.h5')"
```

Success prints `Model loaded successfully!`. That is the real proof the
environment is correct.

---

## 7. Set the data paths for this machine

Each machine reads data from its own hard drive. Open
`main_pipeline_notebooks/main_local.ipynb`, go to **cell 2**, and
comment/uncomment the path block for this machine (there is a template with one
block per machine). Nothing else differs between machines — the Python
environment from steps 3–5 is identical everywhere.

---

## 8. Everyday use (after the first-time setup)

You only do steps 2–5 **once** per machine. Every time you come back:

```bash
conda activate primates
cd primates-sound-detection
# then run the notebook / scripts
```

---

## Troubleshooting (the errors you are likely to hit)

| Symptom | Cause | Fix |
|---|---|---|
| `Unrecognized keyword arguments passed to InputLayer: ['batch_shape', 'optional']` | You're on Keras 2 (TF ≤ 2.15), usually because TF came from `conda install`. | Delete the env (step 3 end) and reinstall via pip (steps 3, 5). Confirm with 6a that Keras is 3.x. |
| `numpy.dtype size changed` / `module compiled against API version` | numpy was installed by conda and clashes with the pip TF wheel. | Same fix: clean env + pip-only install. The pin keeps numpy at 1.26.4. |
| `conda activate` does nothing / no `(primates)` shown | Shell not initialised. | Close and reopen the terminal (Anaconda Prompt on Windows), then `conda activate primates`. |
| `pip` installs into the wrong place | The env wasn't active. | Make sure `(primates)` is showing before `pip install`. Re-run `conda activate primates`. |
| TensorFlow won't import on an old machine (illegal instruction) | Very old CPU without AVX. | Use a newer machine, or ask for a CPU-compatible TF build — note the machine model. |
| The model loads but the **training machine** reported a TF version other than 2.17.0 | Pin doesn't match the machine that produced the model. | See "Aligning with the training machine" below. |

### Aligning with the training machine (do this once)

The pinned TensorFlow (2.17.0) is a known-good Keras-3 version, but the *exact*
version should match the machine that **trained** the model. On the training
machine (e.g. Google Colab) run:

```python
import tensorflow as tf, keras, numpy as np
print("TF:", tf.__version__, "| Keras:", keras.__version__, "| NumPy:", np.__version__)
```

If it reports a different TensorFlow version, edit the one line
`tensorflow==2.17.0` in `requirements-frozen.txt` to match, commit it, and have
every detection machine redo step 5. As long as both sides are **Keras 3**, the
model will load; matching the exact patch version just removes the last bit of
risk.

---

## TL;DR (copy-paste, fresh machine)

```bash
# (install Miniconda first if `conda --version` fails)
conda create -n primates python=3.10 -y
conda activate primates
git clone https://github.com/mo119m/primates-sound-detection.git
cd primates-sound-detection
python -m pip install --upgrade pip
pip install -r requirements-frozen.txt
python -c "import tensorflow as tf, keras; print('TF', tf.__version__, '| Keras', keras.__version__)"
```
