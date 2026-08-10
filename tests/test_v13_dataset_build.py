"""Regression tests for V13 manifest construction safeguards."""

import importlib.util
from pathlib import Path

import pandas as pd


def _build_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_v13_dataset.py"
    spec = importlib.util.spec_from_file_location("build_v13_dataset_test", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_drop_call_like_only_removes_explicit_unreviewed_machine_negatives(tmp_path):
    """A model score must never erase a human-confirmed negative."""
    build = _build_module()
    frame = pd.DataFrame([
        {"path": "birdnet-unreviewed.wav", "label": "Background",
         "source": "birdnet:species", "verified": False},
        {"path": "flagged-unreviewed.wav", "label": "Background",
         "source": "auto_flagged_fp", "verified": "false"},
        {"path": "confirmed-fp.wav", "label": "Background",
         "source": "auto_flagged_fp:confirmed_fp", "verified": True},
        {"path": "reviewed-birdnet.wav", "label": "Background",
         "source": "birdnet:species", "verified": "true"},
        {"path": "reviewed-other.wav", "label": "Background",
         "source": "curated_background", "verified": True},
        {"path": "unreviewed-subtype.wav", "label": "Background",
         "source": "auto_flagged_fp:unknown", "verified": False},
    ])
    scores = tmp_path / "scores.csv"
    pd.DataFrame({
        "path": frame["path"],
        "target_score": [0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
    }).to_csv(scores, index=False)

    kept = build.drop_call_like_negatives(frame, str(scores), threshold=0.5)

    assert set(kept["path"]) == {
        "confirmed-fp.wav",
        "reviewed-birdnet.wav",
        "reviewed-other.wav",
        "unreviewed-subtype.wav",
    }
    assert kept.loc[kept["path"] == "confirmed-fp.wav", "verified"].item() is True
