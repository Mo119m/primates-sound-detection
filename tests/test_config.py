"""Tests for src/config.py -- verify that the configuration module loads
cleanly and exposes the expected constants."""

import os
import sys

# Environment must be set before importing config (see conftest.py).
os.environ.setdefault("PRIMATE_DATA_ROOT", "/tmp/test_data")
os.environ.setdefault("PRIMATE_MODEL_POOLING", "temporal_freqpos")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import config  # noqa: E402


# ------------------------------------------------------------------
# Class structure
# ------------------------------------------------------------------

def test_n_classes_is_four():
    """Four softmax outputs: Cernic, Colobus_guereza, Colobus_confuser, Background."""
    assert config.N_CLASSES == 4


def test_class_names_length():
    """CLASS_NAMES must list exactly N_CLASSES entries."""
    assert len(config.CLASS_NAMES) == 4


def test_class_names_contents():
    """Verify the exact class labels in order."""
    expected = ["Cernic", "Colobus_guereza", "Colobus_confuser", "Background"]
    assert config.CLASS_NAMES == expected


# ------------------------------------------------------------------
# Detection grouping
# ------------------------------------------------------------------

def test_detection_groups_maps_confuser_to_background():
    """The confuser class must fold into the Background detection group."""
    assert config.DETECTION_GROUPS["Colobus_confuser"] == "Background"


def test_detection_groups_covers_all_classes():
    """Every CLASS_NAME must have an entry in DETECTION_GROUPS."""
    for name in config.CLASS_NAMES:
        assert name in config.DETECTION_GROUPS, f"{name} missing from DETECTION_GROUPS"


# ------------------------------------------------------------------
# Low-frequency gate parameters
# ------------------------------------------------------------------

def test_lowfreq_gate_cutoff_exists():
    """LOWFREQ_GATE_CUTOFF should be a positive number (Hz)."""
    assert hasattr(config, "LOWFREQ_GATE_CUTOFF")
    assert config.LOWFREQ_GATE_CUTOFF > 0


def test_lowfreq_gate_threshold_exists():
    """LOWFREQ_GATE_THRESHOLD should be in (0, 1]."""
    assert hasattr(config, "LOWFREQ_GATE_THRESHOLD")
    assert 0 < config.LOWFREQ_GATE_THRESHOLD <= 1.0


# ------------------------------------------------------------------
# Audio / spectrogram sanity
# ------------------------------------------------------------------

def test_sample_rate_is_positive():
    assert config.SAMPLE_RATE > 0


def test_image_dimensions():
    """VGG19 expects 224x224x3."""
    assert config.IMG_HEIGHT == 224
    assert config.IMG_WIDTH == 224
    assert config.IMG_CHANNELS == 3


# ------------------------------------------------------------------
# Data-root resolution
# ------------------------------------------------------------------

def test_data_root_prefers_env_var(monkeypatch):
    monkeypatch.setenv("PRIMATE_DATA_ROOT", "/tmp/some-root")
    assert config._default_data_root() == "/tmp/some-root"


def test_data_root_falls_back_to_repo_data_when_not_on_colab(monkeypatch):
    """Off Colab and with no env var, use the repo's own data/ folder rather
    than the Google Drive path (which would be unwritable locally)."""
    monkeypatch.delenv("PRIMATE_DATA_ROOT", raising=False)
    monkeypatch.setattr(config.os.path, "isdir", lambda p: False)
    root = config._default_data_root()
    assert root == config._REPO_DATA
    assert not root.startswith("/content")


def test_data_root_uses_drive_path_on_colab(monkeypatch):
    monkeypatch.delenv("PRIMATE_DATA_ROOT", raising=False)
    monkeypatch.setattr(config.os.path, "isdir",
                        lambda p: p == "/content/drive/MyDrive")
    assert config._default_data_root() == config._COLAB_ROOT
