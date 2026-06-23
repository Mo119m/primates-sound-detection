# Environment Setup (Windows / macOS / Linux)

From zero to running the notebook. **Run every command one at a time** — copy one
line, paste, press Enter, wait for it to finish, then do the next one.

---

## Step 1 — Install Miniconda

> Skip this step if you already have conda working (type `conda --version` in
> your terminal — if it prints a version number you're good).

1. Go to https://docs.anaconda.com/miniconda/ and download the installer for
   your system (Windows 64-bit / macOS / Linux).
2. Run the installer.
   - **Windows**: when you reach "Advanced Options", **check the box**
     `Add Miniconda3 to my PATH environment variable` (even though it says
     "not recommended" — check it anyway, it saves a lot of trouble).
   - **macOS / Linux**: the defaults are fine.
3. **Close and reopen** your terminal (VS Code, PowerShell, Terminal.app — whatever
   you use). This is required so the terminal can find the new `conda` command.

### First time on a new Miniconda (2025+)

New versions of Miniconda ask you to accept Terms of Service before you can
create environments. If you see a `CondaToSNonInteractiveError`, run these
three commands (one-time):

```
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
```

### Windows: conda not recognized in VS Code?

If VS Code's terminal still says `conda is not recognized` after installing
Miniconda, run this **once**:

```
conda init powershell
```

Then **close VS Code entirely and reopen it**. (Just closing the terminal tab is
not enough — close the whole window.)

---

## Step 2 — Create the Python environment

```
conda create -n primates python=3.12 -y
```

Wait until it prints `done`. Then:

```
conda activate primates
```

**Check**: your prompt should now start with `(primates)`. If it does not,
the activate did not work — see the troubleshooting note in Step 1 above.

> Every time you open a new terminal, you need to `conda activate primates`
> again before running any commands below.

---

## Step 3 — Download the code and open in VS Code

1. Go to the repository page on GitHub.
2. Click the green **Code** button → **Download ZIP**.
3. Unzip it. The ZIP creates a nested folder — open the **inner** folder
   (the one that contains `requirements-frozen.txt`, `src/`, `data/`, etc.).
4. In VS Code: **File → Open Folder** → select that inner folder.
5. Open the VS Code terminal: **Terminal → New Terminal** (or press Ctrl + `).

Your terminal should now show the project folder path. All commands below
assume you are running them in this VS Code terminal.

> Alternatively, if you have git: `git clone https://github.com/mo119m/primates-sound-detection.git`
> then open that folder in VS Code.

---

## Step 4 — Install packages

Make sure your prompt starts with `(primates)`. If not, run
`conda activate primates` first.

```
pip install -r requirements-frozen.txt
pip install jupyter
```

> Use **pip**, not `conda install tensorflow`. The conda version is outdated and
> will not load the model.

If VS Code shows a popup asking "Would you like to create a virtual
environment?" — click **Don't show again**. You already have one (the conda
environment).

---

## Step 5 — Download the pretrained model

The trained V12 model (`best_model_v12.h5`) is not included in the repository
due to file size. Download it and place it at:

```
data/outputs/models/best_model_v12.h5
```

<!-- TODO: replace with actual download link -->
> **Download link:** [to be added — check the README for the latest link]

Create the folder if it doesn't exist yet:

```
mkdir -p data/outputs/models
```

(On Windows: just create the folders manually in File Explorer.)

> If you want to train from scratch instead of using the pretrained model,
> skip this step — the notebook will train automatically when no model is found.

---

## Step 6 — Verify

```
python -c "import tensorflow as tf, keras; print('TF', tf.__version__, '| Keras', keras.__version__)"
```

Expected output: something like `TF 2.20.0 | Keras 3.13.2`. The exact version
depends on your operating system (macOS may install TensorFlow 2.16.x instead
of 2.20.0) — **any TensorFlow 2.16 or newer with Keras 3 is fine** and can
load the model.

If it says `No module named 'tensorflow'`, you are not in the conda
environment — run `conda activate primates` and try again.

---

## Step 7 — Run the notebook

```
jupyter notebook
```

This opens Jupyter in your browser. Navigate to
`main_pipeline_notebooks/main_local.ipynb`, set the data paths for your
machine in **Step 1** of the notebook, then run the cells top to bottom.

---

## Coming back later

Open the project folder in VS Code, then in the terminal:

```
conda activate primates
jupyter notebook
```

---

## Optional: GPU acceleration

The pipeline works on CPU out of the box. GPU makes training faster but is not
required for running detection on a pre-trained model.

| Platform | How to enable GPU |
|----------|-------------------|
| **macOS (Apple Silicon M1–M4)** | `pip install tensorflow-metal` then restart Jupyter |
| **Linux (NVIDIA GPU)** | `pip install tensorflow[and-cuda]` |
| **Windows (NVIDIA GPU)** | Native Windows TF is CPU-only since v2.11. Use [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) + Linux instructions, or just use CPU |
| **No GPU** | Nothing to do — CPU works fine |

The notebook automatically detects GPU on startup and tells you what it found.
