"""Fast contract tests for the V13 trainer's immutable-run safeguards."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _drop_stub_modules():
    """Remove empty stand-ins for tensorflow/keras left in sys.modules.

    Two other test modules install ``types.ModuleType("tensorflow")`` at import
    time so they can exercise auto_cleanup without paying for the real library.
    The stub is never removed, so whether the trainer here can import Keras
    depends on whether those modules were imported first: these three tests pass
    when run alone and fail when run with the suite, with a
    ``ModuleNotFoundError: 'tensorflow' is not a package`` that points nowhere
    near the cause.

    A stub is recognisable by having no ``__file__``. Dropping only those leaves
    a genuine import untouched.
    """
    import sys
    import types
    for name in list(sys.modules):
        root = name.split(".")[0]
        if root not in ("tensorflow", "tensorflow_hub", "keras"):
            continue
        mod = sys.modules.get(name)
        if isinstance(mod, types.ModuleType) and getattr(mod, "__file__", None) is None:
            del sys.modules[name]


def _trainer_module():
    _drop_stub_modules()
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_v13_loso.py"
    spec = importlib.util.spec_from_file_location("train_v13_loso_test", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_valid_artifacts(root):
    manifest = root / "manifest.csv"
    index = root / "index.csv"
    images = root / "images.npy"
    cache = root / "features.npy"
    record = {
        "path": "C:/clips/example.wav",
        "label": "Background",
        "station": "IPA1ST",
        "possible_stations": "IPA1ST",
        "source": "review",
        "verified": True,
        "period": "2021-02",
        "aug": 0,
    }
    pd.DataFrame([record]).to_csv(manifest, index=False)
    pd.DataFrame([{**record, "row": 0, "ok": True}]).to_csv(index, index=False)
    np.save(images, np.zeros((1, 224, 224, 3), dtype=np.uint8))
    np.save(cache, np.zeros((1, 28, 28, 512), dtype=np.float16))
    return manifest, index, images, cache


def test_full_hash_lock_rejects_same_shape_cache_mutation(tmp_path):
    trainer = _trainer_module()
    manifest, index, images, cache = _write_valid_artifacts(tmp_path)
    lock = tmp_path / "artifact.lock.json"
    checker = trainer._checker_module()
    report = checker.run_checks(checker.parse_args([
        "--manifest", str(manifest),
        "--index", str(index),
        "--images", str(images),
        "--cache", str(cache),
        "--full-hash",
        "--write-lock", str(lock),
    ]))
    assert report["ok"]
    receipt = trainer._validate_artifact_lock(
        str(lock), str(manifest), str(index), str(images), str(cache))
    assert receipt["full_hash_verified"] is True

    array = np.load(cache, mmap_mode="r+")
    array[0, 0, 0, 0] = np.float16(1)
    array.flush()
    del array
    with pytest.raises(trainer.ArtifactIntegrityError, match="artifact lock validation failed"):
        trainer._validate_artifact_lock(
            str(lock), str(manifest), str(index), str(images), str(cache))


def test_trainer_rejects_weak_lock_and_interrupted_cache(tmp_path):
    trainer = _trainer_module()
    manifest, index, images, cache = _write_valid_artifacts(tmp_path)
    weak_lock = tmp_path / "weak.lock.json"
    weak_lock.write_text(json.dumps({
        "kind": "v13-artifact-lock",
        "schema_version": 1,
        "fingerprints": {
            "manifest": {"file_sha256": "x", "canonical_rows_sha256": "x"},
            "index": {"file_sha256": "x", "canonical_rows_sha256": "x"},
            "images": {"shape": [1, 224, 224, 3], "dtype": "uint8", "file_bytes": 1},
            "cache": {"shape": [1, 28, 28, 512], "dtype": "float16", "file_bytes": 1},
        },
    }))
    with pytest.raises(trainer.ArtifactIntegrityError, match="not a full-hash receipt"):
        trainer._validate_artifact_lock(
            str(weak_lock), str(manifest), str(index), str(images), str(cache))

    Path(str(cache) + ".partial").write_bytes(b"interrupted")
    with pytest.raises(trainer.ArtifactIntegrityError, match="interrupted artifact"):
        trainer.feature_cache(str(images), str(cache))


def test_output_guard_rejects_existing_outputs_without_overwrite(tmp_path):
    trainer = _trainer_module()
    out = tmp_path / "results.csv"
    metadata = tmp_path / "train.run.json"
    heads = tmp_path / "heads"
    out.write_text("old\n", encoding="utf-8")
    heads.mkdir()
    with pytest.raises(trainer.ArtifactIntegrityError, match="refusing to overwrite"):
        trainer._guard_output_paths(str(out), str(heads), str(metadata), False, False)
