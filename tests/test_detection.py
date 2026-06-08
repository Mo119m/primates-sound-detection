"""Tests for src/detection.py -- detection grouping, probability aggregation,
and the low-frequency gate.

The low-frequency gate uses librosa's STFT, so these tests generate
synthetic audio long enough to fill at least one FFT window (N_FFT=2048
samples at 44100 Hz).
"""

import os
import sys

os.environ.setdefault("PRIMATE_DATA_ROOT", "/tmp/test_data")
os.environ.setdefault("PRIMATE_MODEL_POOLING", "temporal_freq")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import config  # noqa: E402
import detection  # noqa: E402


# ---- helpers --------------------------------------------------------

def _sine(freq_hz: float, duration: float = 2.0,
          sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """Generate a pure sine wave at *freq_hz*."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)


# ---- get_detection_groups -------------------------------------------

def test_get_detection_groups_labels():
    """Labels should contain Cernic, Colobus_guereza, and Background (last)."""
    labels, _ = detection.get_detection_groups()
    assert "Cernic" in labels
    assert "Colobus_guereza" in labels
    assert labels[-1] == "Background", "Background must be the last label"


def test_get_detection_groups_confuser_merged():
    """Colobus_confuser should NOT appear as its own group label."""
    labels, _ = detection.get_detection_groups()
    assert "Colobus_confuser" not in labels


def test_get_detection_groups_indices_cover_all_classes():
    """Every CLASS_NAMES index must appear in exactly one group."""
    _, indices = detection.get_detection_groups()
    all_idx = sorted(i for idxs in indices.values() for i in idxs)
    assert all_idx == list(range(len(config.CLASS_NAMES)))


# ---- group_probabilities -------------------------------------------

def test_group_probabilities_sums_to_one():
    """Grouped probabilities of a valid softmax vector must still sum to 1."""
    labels, indices = detection.get_detection_groups()
    pred = np.array([0.1, 0.3, 0.2, 0.4])  # Cernic, Colobus, Confuser, Bg
    grouped = detection.group_probabilities(pred, labels, indices)
    assert abs(grouped.sum() - 1.0) < 1e-6


def test_group_probabilities_confuser_adds_to_background():
    """Confuser probability must merge into the Background group."""
    labels, indices = detection.get_detection_groups()
    # All mass on Colobus_confuser (index 2) and Background (index 3)
    pred = np.array([0.0, 0.0, 0.3, 0.7])
    grouped = detection.group_probabilities(pred, labels, indices)
    bg_idx = labels.index("Background")
    assert abs(grouped[bg_idx] - 1.0) < 1e-6, (
        "Confuser + Background should sum to 1.0 in the Background group"
    )


def test_group_probabilities_species_passthrough():
    """Cernic and Colobus_guereza each map to their own group unchanged."""
    labels, indices = detection.get_detection_groups()
    pred = np.array([0.5, 0.3, 0.1, 0.1])
    grouped = detection.group_probabilities(pred, labels, indices)
    cernic_idx = labels.index("Cernic")
    colobus_idx = labels.index("Colobus_guereza")
    assert abs(grouped[cernic_idx] - 0.5) < 1e-6
    assert abs(grouped[colobus_idx] - 0.3) < 1e-6


# ---- lowfreq_energy_ratio ------------------------------------------

def test_lowfreq_ratio_high_for_low_sine():
    """A 300 Hz sine should have nearly all energy below 1 kHz."""
    audio = _sine(300.0)
    ratio = detection.lowfreq_energy_ratio(audio, cutoff=1000)
    assert ratio is not None
    assert ratio > 0.95, f"Expected ratio > 0.95 for 300 Hz sine, got {ratio:.4f}"


def test_lowfreq_ratio_low_for_high_sine():
    """A 5 kHz sine should have almost no energy below 1 kHz."""
    audio = _sine(5000.0)
    ratio = detection.lowfreq_energy_ratio(audio, cutoff=1000)
    assert ratio is not None
    assert ratio < 0.05, f"Expected ratio < 0.05 for 5 kHz sine, got {ratio:.4f}"


def test_lowfreq_ratio_short_audio_returns_none():
    """Audio shorter than N_FFT should return None (not enough data for STFT)."""
    short = np.zeros(config.N_FFT - 1, dtype=np.float32)
    ratio = detection.lowfreq_energy_ratio(short)
    assert ratio is None


# ---- apply_lowfreq_gate --------------------------------------------
#
# The gate logic: real Colobus guereza roars are low-frequency (ratio ~ 0.98).
# False positives (insects, cicadas) are high-frequency (ratio ~ 0.01).
# A Colobus detection is KEPT when ratio >= threshold, DROPPED otherwise.
# Non-Colobus species pass through unconditionally.

def _detections_df(species_list, sr=config.SAMPLE_RATE, window=2.0):
    """Build a minimal detections DataFrame."""
    rows = []
    for i, sp in enumerate(species_list):
        rows.append({
            "start_time": i * window,
            "end_time": (i + 1) * window,
            "species": sp,
            "confidence": 0.9,
        })
    return pd.DataFrame(rows)


def _concat_audio(signals):
    """Concatenate a list of waveform arrays into one recording."""
    return np.concatenate(signals)


def test_lowfreq_gate_drops_highfreq_colobus():
    """A Colobus detection backed by a 5 kHz sine (insect-like) should be dropped."""
    df = _detections_df(["Colobus_guereza"])
    audio = _sine(5000.0, duration=2.0)
    result = detection.apply_lowfreq_gate(
        df, audio, sr=config.SAMPLE_RATE, threshold=0.5,
    )
    assert len(result) == 0, "High-freq Colobus detection should be suppressed"


def test_lowfreq_gate_keeps_lowfreq_colobus():
    """A Colobus detection backed by a 300 Hz sine (real roar-like) should pass."""
    df = _detections_df(["Colobus_guereza"])
    audio = _sine(300.0, duration=2.0)
    result = detection.apply_lowfreq_gate(
        df, audio, sr=config.SAMPLE_RATE, threshold=0.5,
    )
    assert len(result) == 1


def test_lowfreq_gate_passes_cernic_unconditionally():
    """Cernic detections are not the target species -- they always pass."""
    df = _detections_df(["Cernic"])
    audio = _sine(5000.0, duration=2.0)  # high-freq, but irrelevant for Cernic
    result = detection.apply_lowfreq_gate(
        df, audio, sr=config.SAMPLE_RATE, threshold=0.5,
    )
    assert len(result) == 1, "Cernic detections must never be gated"


def test_lowfreq_gate_mixed_batch():
    """Batch: Cernic (high-freq) survives, Colobus (high-freq) is dropped."""
    df = _detections_df(["Cernic", "Colobus_guereza"])
    audio = _concat_audio([_sine(5000.0, duration=2.0),
                           _sine(5000.0, duration=2.0)])
    result = detection.apply_lowfreq_gate(
        df, audio, sr=config.SAMPLE_RATE, threshold=0.5,
    )
    assert len(result) == 1
    assert result.iloc[0]["species"] == "Cernic"
