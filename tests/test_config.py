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

def test_n_classes_matches_species_folders():
    """N_CLASSES is the species count plus Background, and must stay derived."""
    assert config.N_CLASSES == len(config.SPECIES_FOLDERS) + 1


def test_class_names_length():
    """CLASS_NAMES must list exactly N_CLASSES entries, with no duplicates."""
    assert len(config.CLASS_NAMES) == config.N_CLASSES
    assert len(set(config.CLASS_NAMES)) == config.N_CLASSES


def test_class_names_contents():
    """The five softmax outputs, in the order the deployed model was trained in.

    Order is load-bearing, not cosmetic: src/detection.py reads a softmax vector
    positionally against this list. The V13 leave-one-station-out sweep trains
    against sorted(labels) instead, which is a DIFFERENT order, so a head from
    that sweep must be permuted into this order before any detection code sees
    it (scripts/assemble_fold_model.py does that and refuses to save a model
    that fails the check).
    """
    expected = ["Cernic", "Colobus_guereza", "Colobus_confuser",
                "C_pogonias", "Background"]
    assert config.CLASS_NAMES == expected


def test_background_is_last():
    """Background must remain the final index; several scripts assume it."""
    assert config.CLASS_NAMES[-1] == "Background"


def test_every_class_has_a_detection_group():
    """A class with no entry in DETECTION_GROUPS silently becomes its own group."""
    assert set(config.DETECTION_GROUPS) == set(config.CLASS_NAMES)
    assert set(config.DETECTION_GROUPS.values()) <= set(config.CLASS_NAMES)


def test_embed_snr_range_is_sane():
    """The embedding level is measured from the field clips; keep it plausible."""
    lo, hi = config.EMBED_SNR_DB_RANGE
    assert lo < hi
    assert -30.0 <= lo and hi <= 30.0


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


# Measured by scripts/calibrate_colobus_gate.py over the 253 C. guereza
# detections the 16-station deployment produced, none of which manual listening
# found to be a genuine roar.
FIELD_FP_MEDIAN_LOWFREQ_RATIO = 0.396


def test_lowfreq_gate_threshold_excludes_most_field_false_positives():
    """
    The gate must sit above the median of the negatives it actually meets.

    The original 0.20 was calibrated against Colobus_confuser clips, which top
    out at 0.092, and it separated them perfectly. It was never tested against
    thunder, which is itself low-frequency: at 0.20 nearly nine in ten of the
    field detections passed. A threshold below the field median means more than
    half of a population known to be entirely false is admitted, which is the
    specific failure this bound exists to prevent -- not a style rule.

    Lowering the threshold is allowed, but only together with evidence about a
    different negative population, which means updating the constant above.
    """
    assert config.LOWFREQ_GATE_THRESHOLD > FIELD_FP_MEDIAN_LOWFREQ_RATIO, (
        f"threshold {config.LOWFREQ_GATE_THRESHOLD} admits over half the field "
        f"detections (median ratio {FIELD_FP_MEDIAN_LOWFREQ_RATIO})"
    )


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
