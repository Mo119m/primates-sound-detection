"""Tests for src/preprocessing.py -- mel-spectrogram conversion and the
full audio-to-RGB preprocessing pipeline.

All tests use numpy-generated synthetic audio (sine waves) so no real
WAV files are needed.
"""

import os
import sys

os.environ.setdefault("PRIMATE_DATA_ROOT", "/tmp/test_data")
os.environ.setdefault("PRIMATE_MODEL_POOLING", "temporal_freq")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import config  # noqa: E402
import preprocessing  # noqa: E402


# ---- helpers --------------------------------------------------------

def _sine(freq_hz: float = 440.0, duration: float = 2.0,
          sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """Generate a pure sine wave at *freq_hz*."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)


# ---- audio_to_melspectrogram ----------------------------------------

def test_melspec_output_is_2d():
    """Mel-spectrogram of a mono waveform must be a 2-D array."""
    audio = _sine()
    mel = preprocessing.audio_to_melspectrogram(audio, sr=config.SAMPLE_RATE)
    assert mel.ndim == 2


def test_melspec_n_mels_rows():
    """First axis must equal N_MELS (128 mel bands)."""
    audio = _sine()
    mel = preprocessing.audio_to_melspectrogram(audio, sr=config.SAMPLE_RATE)
    assert mel.shape[0] == config.N_MELS


def test_melspec_time_frames():
    """Number of time frames should match ceil(n_samples / HOP_LENGTH) + 1."""
    audio = _sine(duration=config.CLIP_DURATION)
    mel = preprocessing.audio_to_melspectrogram(audio, sr=config.SAMPLE_RATE)
    expected_frames = 1 + int(len(audio) // config.HOP_LENGTH)
    # librosa can differ by 1 depending on centering; allow a margin
    assert abs(mel.shape[1] - expected_frames) <= 1


def test_melspec_values_in_db():
    """Power-to-dB output should contain non-positive values (ref=max)."""
    audio = _sine()
    mel = preprocessing.audio_to_melspectrogram(audio, sr=config.SAMPLE_RATE)
    assert mel.max() <= 0.0, "dB scale with ref=max should peak at 0"
    assert mel.min() < 0.0, "Should contain negative dB values"


# ---- preprocess_audio (full pipeline) -------------------------------

def test_preprocess_audio_output_shape():
    """Full pipeline must produce a (224, 224, 3) uint8 RGB image."""
    audio = _sine(duration=config.CLIP_DURATION)
    img = preprocessing.preprocess_audio(audio, sr=config.SAMPLE_RATE)
    assert img.shape == (config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)


def test_preprocess_audio_dtype():
    """Output of preprocess_audio should be uint8 (0-255 range)."""
    audio = _sine(duration=config.CLIP_DURATION)
    img = preprocessing.preprocess_audio(audio, sr=config.SAMPLE_RATE)
    assert img.dtype == np.uint8


def test_preprocess_audio_pixel_range():
    """Pixel values must lie in [0, 255]."""
    audio = _sine(duration=config.CLIP_DURATION)
    img = preprocessing.preprocess_audio(audio, sr=config.SAMPLE_RATE)
    assert img.min() >= 0
    assert img.max() <= 255


def test_preprocess_audio_rgb_channels_identical():
    """The three RGB channels should be identical (greyscale copy)."""
    audio = _sine(duration=config.CLIP_DURATION)
    img = preprocessing.preprocess_audio(audio, sr=config.SAMPLE_RATE)
    np.testing.assert_array_equal(img[:, :, 0], img[:, :, 1])
    np.testing.assert_array_equal(img[:, :, 1], img[:, :, 2])


# ---- preprocess_for_model -------------------------------------------

def test_preprocess_for_model_range():
    """Model input should be float32 in [0, 1]."""
    audio = _sine(duration=config.CLIP_DURATION)
    img = preprocessing.preprocess_audio(audio, sr=config.SAMPLE_RATE)
    out = preprocessing.preprocess_for_model(img)
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0


# ---- edge cases -----------------------------------------------------

def test_preprocess_short_audio():
    """Audio shorter than CLIP_DURATION should still produce (224,224,3)."""
    short = _sine(duration=0.5)  # 0.5 s instead of 2.0 s
    img = preprocessing.preprocess_audio(short, sr=config.SAMPLE_RATE)
    assert img.shape == (config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)


def test_preprocess_long_audio():
    """Audio longer than CLIP_DURATION should still produce (224,224,3)."""
    long = _sine(duration=5.0)
    img = preprocessing.preprocess_audio(long, sr=config.SAMPLE_RATE)
    assert img.shape == (config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)
