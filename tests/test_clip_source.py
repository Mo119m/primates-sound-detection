"""Tests for reading detection audio from the already-exported review clips
instead of re-cutting it from the source recordings (auto_cleanup.
load_clips_from_dir).

auto_cleanup imports the model module, which pulls in TensorFlow; these tests
stub it so the clip-slicing logic can be tested on its own.
"""

import os
import sys
import types
import tempfile

os.environ.setdefault("PRIMATE_DATA_ROOT", "/tmp/test_data")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

for _name in ("tensorflow", "tensorflow_hub"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
_stub = types.ModuleType("model")
_stub.load_trained_model = lambda *a, **k: None
sys.modules.setdefault("model", _stub)

import config  # noqa: E402
import auto_cleanup  # noqa: E402

sf = pytest.importorskip("soundfile")

PAD = 0.5


def _fixture(det_starts, recording="recX", species="Cernic"):
    """Write a synthetic recording and its exported clips; return (y, dir, df)."""
    sr, win = config.SAMPLE_RATE, config.WINDOW_SIZE
    rng = np.random.default_rng(0)
    y = (rng.standard_normal(sr * 60) * 0.05).astype("float32")
    for i, s in enumerate(det_starts):
        a = int(s * sr)
        y[a:a + int(win * sr)] += 0.2 * (i + 1)

    clips_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(clips_dir, species, "IPA1ST"), exist_ok=True)
    rows = []
    for s in det_starts:
        a = int(max(0, s - PAD) * sr)
        b = int(min(len(y), s + win + PAD) * sr)
        name = f"{species}__{recording}__{int(s):05d}s__conf0.900.wav"
        sf.write(os.path.join(clips_dir, species, "IPA1ST", name),
                 y[a:b], sr, subtype="FLOAT")
        rows.append(dict(species=species, source_file=f"{recording}.wav",
                         start_time=s))
    return y, clips_dir, pd.DataFrame(rows)


def test_exported_clip_reproduces_the_analysis_window():
    """The window read back from a padded clip must equal the window that would
    have been cut from the source recording -- including at start_time 0, where
    the exporter could not pad on the left."""
    starts = [10.0, 25.0, 0.0]
    y, clips_dir, det = _fixture(starts)
    clips = auto_cleanup.load_clips_from_dir(det, clips_dir, padding=PAD,
                                             verbose=False)
    sr, win = config.SAMPLE_RATE, config.WINDOW_SIZE
    assert len(clips) == len(starts)
    for clip, s in zip(clips, starts):
        expected = y[int(s * sr):int(s * sr) + int(win * sr)]
        assert len(clip) == int(win * sr)
        assert np.max(np.abs(clip - expected)) < 1e-6


def test_missing_clips_are_reported_not_silently_zeroed():
    _, clips_dir, det = _fixture([10.0])
    det = pd.concat([det, pd.DataFrame([dict(species="Cernic",
                                             source_file="recX.wav",
                                             start_time=999.0)])],
                    ignore_index=True)
    clips = auto_cleanup.load_clips_from_dir(det, clips_dir, padding=PAD,
                                             verbose=False)
    assert len(clips) == 2
    assert np.all(clips[1] == 0)          # the unmatched one is silence
    assert np.any(clips[0] != 0)          # the matched one is real audio


def test_all_missing_raises():
    _, clips_dir, det = _fixture([10.0])
    det["source_file"] = "other_recording.wav"
    with pytest.raises(FileNotFoundError, match="matched an exported clip"):
        auto_cleanup.load_clips_from_dir(det, clips_dir, padding=PAD,
                                         verbose=False)
