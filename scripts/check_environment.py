"""
Verify that the pipeline is correctly configured to run locally.

Checks, in order:
  1. Required Python packages import.
  2. The PRIMATE_* environment variables / resolved config paths.
  3. That the data directories exist (species/, background/, field_recordings).
  4. That the production pooling head is selected for the V10 model.

Run from the repo root after setting your environment variables, e.g.:

    set -a; source .env; set +a
    python scripts/check_environment.py

Exit code is 0 if everything required is present, 1 otherwise. Missing
optional pieces (field_recordings, YAMNet deps) produce warnings, not errors.
"""

import importlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

OK = "[ OK ]"
WARN = "[WARN]"
FAIL = "[FAIL]"


def check_packages():
    """Required and optional package imports."""
    required = ['numpy', 'pandas', 'librosa', 'soundfile', 'sklearn',
                'matplotlib', 'cv2', 'tensorflow']
    optional = ['tensorflow_hub', 'resampy']  # YAMNet auto-cleanup filter
    errors = 0

    print("Packages")
    for pkg in required:
        try:
            importlib.import_module(pkg)
            print(f"  {OK} {pkg}")
        except ImportError:
            print(f"  {FAIL} {pkg} -- pip install -r requirements.txt")
            errors += 1
    for pkg in optional:
        try:
            importlib.import_module(pkg)
            print(f"  {OK} {pkg} (optional, YAMNet filter)")
        except ImportError:
            print(f"  {WARN} {pkg} missing -- only needed for auto_cleanup's "
                  f"YAMNet filter")
    return errors


def check_config():
    """Resolved config paths and the data layout they point at."""
    import config

    errors = 0
    print("\nConfiguration")
    print(f"  DATA_ROOT       = {config.DRIVE_ROOT}")
    print(f"  AUDIO_ROOT      = {config.AUDIO_ROOT}")
    print(f"  IPA_ROOT        = {config.IPA_ROOT}")
    print(f"  OUTPUT_ROOT     = {config.OUTPUT_ROOT}")
    print(f"  MODEL_POOLING   = {config.MODEL_POOLING}")
    print(f"  N_CLASSES       = {config.N_CLASSES} ({', '.join(config.CLASS_NAMES)})")

    # Did the user actually point us away from the Colab default?
    if config.DRIVE_ROOT == "/content/drive/MyDrive/primates-data":
        print(f"  {WARN} DATA_ROOT is still the Colab default. "
              f"Set PRIMATE_DATA_ROOT to your local path.")

    # Required data directories.
    species = os.path.join(config.AUDIO_ROOT, 'species')
    background = os.path.join(config.AUDIO_ROOT, 'background')
    for label, path in [('species/', species), ('background/', background)]:
        if os.path.isdir(path):
            print(f"  {OK} {label} found at {path}")
        else:
            print(f"  {FAIL} {label} not found at {path}")
            errors += 1

    # Field recordings are optional (only needed for the detection stage).
    if os.path.isdir(config.IPA_ROOT):
        print(f"  {OK} field recordings found at {config.IPA_ROOT}")
    else:
        print(f"  {WARN} field recordings not found at {config.IPA_ROOT} "
              f"(only needed for the detection stage)")

    # Production head reminder.
    if config.MODEL_POOLING != 'temporal_freq':
        print(f"  {WARN} MODEL_POOLING is '{config.MODEL_POOLING}'. "
              f"Set PRIMATE_MODEL_POOLING=temporal_freq to reproduce V10.")

    return errors


def main():
    errors = check_packages()
    try:
        errors += check_config()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"\n{FAIL} could not import config: {exc}")
        errors += 1

    print()
    if errors:
        print(f"{FAIL} {errors} required check(s) failed -- see above.")
        sys.exit(1)
    print(f"{OK} Environment looks ready to run locally.")
    sys.exit(0)


if __name__ == '__main__':
    main()
