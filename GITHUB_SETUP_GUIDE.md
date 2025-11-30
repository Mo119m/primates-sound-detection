# GitHub Repository Setup Instructions

This document provides step-by-step instructions for organizing your primate vocalization detection project for GitHub.

## Created Files

I have created the following new files for your GitHub repository:

### Configuration Files
1. `.gitignore` - Specifies which files Git should ignore
2. `requirements.txt` - Lists all Python dependencies
3. `setup.py` - Enables pip installation of the package
4. `LICENSE` - MIT License for open source distribution

### Documentation Files
5. `README.md` - Main project documentation (updated with GitHub-specific content)
6. `CONTRIBUTING.md` - Guidelines for contributors
7. `CHANGELOG.md` - Version history and planned features
8. `docs/QUICK_START.md` - Quick start guide (updated, no decorative symbols)

### Package Structure Files
9. `src/__init__.py` - Makes src directory a Python package
10. `examples/README.md` - Explains how to obtain sample data

## Complete File Organization

Organize your files according to this structure:

```
primate-vocalization-detection/
│
├── .gitignore
├── LICENSE
├── README.md
├── CONTRIBUTING.md
├── CHANGELOG.md
├── requirements.txt
├── setup.py
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── augmentation.py
│   ├── model.py
│   ├── train.py
│   ├── detection.py
│   └── utils.py
│
├── scripts/
│   └── run_hard_negative_mining.py
│
├── notebooks/
│   └── main_pipeline.ipynb
│
├── docs/
│   ├── QUICK_START.md
│   ├── FILE_MANIFEST.md
│   └── SETUP_TUTORIAL.md
│
├── examples/
│   └── README.md
│
└── tests/
    └── __init__.py (create empty file for future tests)
```

## Step-by-Step Setup Process

### Step 1: Create Local Repository Structure

1. Create a new directory for your project:
```bash
mkdir primate-vocalization-detection
cd primate-vocalization-detection
```

2. Create subdirectories:
```bash
mkdir src
mkdir scripts
mkdir notebooks
mkdir docs
mkdir examples
mkdir tests
```

### Step 2: Move Your Existing Files

Move your existing Python files to appropriate directories:

From your current location to new structure:
- `config.py` → `src/config.py`
- `data_loader.py` → `src/data_loader.py`
- `preprocessing.py` → `src/preprocessing.py`
- `augmentation.py` → `src/augmentation.py`
- `model.py` → `src/model.py`
- `train.py` → `src/train.py`
- `detection.py` → `src/detection.py`
- `utils.py` → `src/utils.py`
- `run_hard_negative_mining.py` → `scripts/run_hard_negative_mining.py`
- `main_pipeline.ipynb` → `notebooks/main_pipeline.ipynb`
- `QUICK_START.md` → `docs/QUICK_START.md` (use updated version)
- `FILE_MANIFEST.md` → `docs/FILE_MANIFEST.md`
- `SETUP_TUTORIAL.md` → `docs/SETUP_TUTORIAL.md`

### Step 3: Add New Files

Copy the files I created into your project:
- `.gitignore` → root directory
- `LICENSE` → root directory
- `README.md` → root directory (updated version)
- `CONTRIBUTING.md` → root directory
- `CHANGELOG.md` → root directory
- `requirements.txt` → root directory
- `setup.py` → root directory
- `src/__init__.py` → `src/` directory
- `examples/README.md` → `examples/` directory

Create an empty `__init__.py` in tests directory:
```bash
touch tests/__init__.py
```

### Step 4: Update Import Statements

After reorganizing files, you need to update import statements in your code.

#### In notebooks/main_pipeline.ipynb

Add at the beginning:
```python
import sys
sys.path.append('../src')
```

Or use absolute imports:
```python
from src import config
from src import data_loader
# etc.
```

#### In scripts/run_hard_negative_mining.py

Update imports:
```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src import config
from src import preprocessing
from src import data_loader
from src import model as model_module
```

### Step 5: Initialize Git Repository

```bash
cd primate-vocalization-detection
git init
git add .
git commit -m "Initial commit: Complete pipeline for primate vocalization detection"
```

### Step 6: Create GitHub Repository

1. Go to https://github.com
2. Click "New repository"
3. Name it: `primate-vocalization-detection`
4. Do NOT initialize with README, .gitignore, or license (you already have these)
5. Click "Create repository"

### Step 7: Connect Local to GitHub

```bash
git remote add origin https://github.com/YOUR_USERNAME/primate-vocalization-detection.git
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

## Important Notes

### Files NOT to Upload

The `.gitignore` file ensures these are excluded:
- Audio files (*.wav, *.mp3)
- Model files (*.h5, *.hdf5)
- Output directories (outputs/, detections/, visualizations/)
- Google Drive paths (drive/)
- Python cache files (__pycache__/)

### Sensitive Information

Before uploading, verify that:
- No personal Google Drive paths are hardcoded
- No API keys or credentials are included
- No private research data is committed

### Config.py Modifications

You may want to modify `src/config.py` to use environment variables or make paths more configurable:

```python
import os

# Allow override via environment variable
DRIVE_ROOT = os.getenv('DRIVE_ROOT', "/content/drive/MyDrive/chimp-audio")
AUDIO_ROOT = os.path.join(DRIVE_ROOT, "audio")
```

## Testing the Structure

After setup, test that imports work correctly:

```python
# In Python interpreter or notebook
import sys
sys.path.append('src')

from src import config
from src import data_loader

print("Import successful!")
config.print_config_summary()
```

## Optional: Package Installation

Users can install your package directly from GitHub:

```bash
pip install git+https://github.com/YOUR_USERNAME/primate-vocalization-detection.git
```

Or for local development:
```bash
cd primate-vocalization-detection
pip install -e .
```

This enables imports like:
```python
from src import config
from src import train
```

## Repository Customization

### Update Personal Information

Edit these files to add your information:
1. `LICENSE` - Replace "Mo" with your full name
2. `README.md` - Update author information, contact details
3. `setup.py` - Add your email and GitHub username
4. `CITATION.cff` (optional) - Create for proper academic citation

### Add Repository Description

On GitHub repository page:
- Click "About" settings
- Add description: "Automated detection of primate vocalizations in rainforest recordings using VGG19 transfer learning"
- Add topics: `deep-learning`, `bioacoustics`, `primate-vocalizations`, `transfer-learning`, `audio-classification`

### Create Release

After confirming everything works:
1. Go to "Releases" on GitHub
2. Click "Create a new release"
3. Tag version: `v1.0.0`
4. Release title: "Initial Release - Complete Pipeline"
5. Describe features and known limitations

## Collaboration Setup

### Branch Protection

For collaborative work:
1. Go to Settings → Branches
2. Add branch protection rule for `main`
3. Require pull request reviews
4. Require status checks to pass

### Issue Templates

Create `.github/ISSUE_TEMPLATE/` directory with templates for:
- Bug reports
- Feature requests
- Questions

## Documentation Website (Optional)

Consider using GitHub Pages for documentation:
1. Create `docs/` branch or use main branch `docs/` folder
2. Enable GitHub Pages in Settings
3. Choose theme
4. Documentation will be available at: `https://YOUR_USERNAME.github.io/primate-vocalization-detection/`

## Continuous Integration (Future)

Consider adding GitHub Actions for:
- Automated testing
- Code quality checks (flake8, black)
- Documentation building

Create `.github/workflows/tests.yml` for CI/CD.

## Summary Checklist

- [ ] Created all directory structure
- [ ] Moved existing files to appropriate locations
- [ ] Added new configuration files (.gitignore, requirements.txt, etc.)
- [ ] Updated import statements in code
- [ ] Initialized Git repository
- [ ] Created GitHub repository
- [ ] Pushed code to GitHub
- [ ] Verified .gitignore is working (no large files uploaded)
- [ ] Updated README with your information
- [ ] Added repository description and topics on GitHub
- [ ] Tested that imports work correctly
- [ ] Created first release (optional)

## Getting Help

If you encounter issues:
1. Verify directory structure matches exactly
2. Check that all files are in correct locations
3. Confirm import statements are updated
4. Review Git status: `git status`
5. Check what will be committed: `git diff`

## Next Steps After GitHub Setup

1. Share repository link with collaborators (Santiago, Professor Claudia)
2. Add collaborators on GitHub if needed
3. Create project roadmap in GitHub Projects
4. Set up issue tracking for known problems
5. Document hard negative mining workflow progress
6. Consider writing academic paper describing methodology
