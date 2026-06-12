"""
Shared fixtures for the primate-detection test suite.

Environment variables are set *before* any src module is imported so that
config.py resolves paths to a harmless temp directory instead of the real
Google Drive / data root.
"""

import os
import sys

# ---------------------------------------------------------------------------
# Environment – MUST happen before any src import
# ---------------------------------------------------------------------------
os.environ.setdefault("PRIMATE_DATA_ROOT", "/tmp/test_data")
os.environ.setdefault("PRIMATE_MODEL_POOLING", "temporal_freqpos")

# ---------------------------------------------------------------------------
# Path – make ``import config`` (etc.) work from tests/
# ---------------------------------------------------------------------------
_SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
