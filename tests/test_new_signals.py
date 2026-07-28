"""Tests for the extra per-detection signals in auto_cleanup: the softmax
margin (which keeps varying where the top probability has saturated) and the
station recurrence measure (which sees repetitive noise the Mahalanobis filter
cannot).

auto_cleanup imports the model module, which pulls in TensorFlow; it is stubbed
so these numeric helpers can be tested on their own.
"""

import os
import sys
import types

os.environ.setdefault("PRIMATE_DATA_ROOT", "/tmp/test_data")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

for _name in ("tensorflow", "tensorflow_hub"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
_stub = types.ModuleType("model")
_stub.load_trained_model = lambda *a, **k: None
sys.modules.setdefault("model", _stub)

import auto_cleanup  # noqa: E402


def test_margin_separates_what_confidence_cannot():
    """Two detections can share a high top probability while differing in how
    close the runner-up is; the margin is what distinguishes them."""
    probs = np.array([
        [0.990, 0.005, 0.003, 0.002],   # decisive
        [0.500, 0.490, 0.006, 0.004],   # ambiguous
    ])
    det = pd.DataFrame([{"species": "Cernic"}, {"species": "Cernic"}])
    out = auto_cleanup.annotate_softmax_margin(det, probs)

    assert abs(out["softmax_margin"][0] - 0.985) < 1e-9
    assert abs(out["softmax_margin"][1] - 0.010) < 1e-9
    # the ambiguous window also carries more entropy
    assert out["softmax_entropy"][1] > out["softmax_entropy"][0]


def test_margin_is_nan_when_probabilities_are_unavailable():
    det = pd.DataFrame([{"species": "Cernic"}] * 3)
    out = auto_cleanup.annotate_softmax_margin(det, np.zeros((2, 4)))
    assert out["softmax_margin"].isna().all()


def test_recurrence_flags_repetitive_detections():
    """Near-identical detections at one station must score far tighter than
    detections that vary between utterances."""
    rng = np.random.default_rng(0)
    repetitive = np.tile(np.array([1.0, 0.0, 0.0]), (8, 1)) + rng.normal(0, 1e-3, (8, 3))
    varied = rng.normal(0, 3.0, (8, 3)) + np.array([20.0, 0.0, 0.0])
    feats = np.vstack([repetitive, varied]).astype("float32")
    det = pd.DataFrame([{"station": "IPA4ST", "species": "Cernic"}] * 16)

    out = auto_cleanup.annotate_station_recurrence(det, feats, k=3, verbose=False)
    d = out["recurrence_knn_dist"].to_numpy()
    assert d[:8].max() < d[8:].min()


def test_recurrence_is_computed_within_station_not_across():
    """A station whose detections are all distinct must not look repetitive
    just because another station has a tight cluster."""
    tight = np.tile(np.array([0.0, 0.0]), (6, 1)).astype("float32")
    spread = (np.arange(6, dtype="float32")[:, None] * 10.0).repeat(2, axis=1)
    feats = np.vstack([tight, spread])
    det = pd.DataFrame([{"station": "A", "species": "Cernic"}] * 6 +
                       [{"station": "B", "species": "Cernic"}] * 6)

    out = auto_cleanup.annotate_station_recurrence(det, feats, k=2, verbose=False)
    d = out["recurrence_knn_dist"].to_numpy()
    assert np.nanmax(d[:6]) == 0.0        # station A is repetitive
    assert np.nanmin(d[6:]) > 0.0         # station B is not
