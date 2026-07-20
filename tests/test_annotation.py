"""Tests for src/annotation.py -- clip scanning, resumable labels, and the
field-deployment summary. These use empty placeholder .wav files because the
scanner only parses filenames/paths; no audio decoding is involved.
"""

import os
import sys
import tempfile

os.environ.setdefault("PRIMATE_DATA_ROOT", "/tmp/test_data")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import annotation  # noqa: E402


def _make_tree():
    """Create a detected_clips tree and return its path."""
    root = tempfile.mkdtemp()
    files = [
        "Cernic/IPA1ST/Cernic__recA__00012s__conf0.812.wav",
        "Cernic/IPA1ST/Cernic__recA__00045s__conf0.910.wav",
        "Cernic/IPA2ST/Cernic__recB__00003s__conf0.640.wav",
        "Colobus_guereza/IPA1ST/Colobus_guereza__recA__00099s__conf0.755.wav",
        "Colobus_guereza/IPA2ST/Colobus_guereza__recB__00120s__conf0.880.wav",
    ]
    for rel in files:
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").close()
    return root


def test_scan_parses_metadata():
    df = annotation.scan_clips(_make_tree())
    assert len(df) == 5
    assert set(df["species"]) == {"Cernic", "Colobus_guereza"}
    assert set(df["station"]) == {"IPA1ST", "IPA2ST"}
    row = df[df["start_s"] == 12].iloc[0]
    assert row["species"] == "Cernic"
    assert abs(row["confidence"] - 0.812) < 1e-9


def test_labels_are_resumable_and_upsert():
    df = annotation.scan_clips(_make_tree())
    csv = os.path.join(tempfile.mkdtemp(), "labels.csv")
    cid = df.iloc[0]["clip_id"]
    annotation.save_annotation(cid, annotation.TRUE_CALL, csv_path=csv)
    annotation.save_annotation(cid, annotation.FALSE_POS, csv_path=csv)  # re-label
    ann = annotation.load_annotations(csv)
    assert len(ann) == 1                       # upsert, not duplicate
    # labels must survive a round-trip as strings, not booleans
    val = ann.iloc[0]["label"]
    assert val == annotation.FALSE_POS
    assert isinstance(val, str) and not isinstance(val, bool)


def test_summary_counts_and_precision():
    df = annotation.scan_clips(_make_tree())
    csv = os.path.join(tempfile.mkdtemp(), "labels.csv")
    ids = list(df["clip_id"])
    cernic = df[df["species"] == "Cernic"]["clip_id"].tolist()
    colobus = df[df["species"] == "Colobus_guereza"]["clip_id"].tolist()
    annotation.save_annotation(cernic[0], annotation.TRUE_CALL, csv_path=csv)
    annotation.save_annotation(cernic[1], annotation.TRUE_CALL, csv_path=csv)
    annotation.save_annotation(cernic[2], annotation.FALSE_POS, csv_path=csv)
    annotation.save_annotation(colobus[0], annotation.TRUE_CALL, csv_path=csv)
    annotation.save_annotation(colobus[1], annotation.FALSE_POS, csv_path=csv)

    ann = annotation.load_annotations(csv)
    assert annotation.progress(df, ann) == (5, 5)

    s = annotation.summarize(df, ann)
    ps = s["per_species"]
    assert int(ps.loc["Cernic", "true"]) == 2
    assert int(ps.loc["Cernic", "false"]) == 1
    assert abs(float(ps.loc["Cernic", "precision"]) - (2 / 3)) < 1e-4
    assert int(ps.loc["Colobus_guereza", "true"]) == 1
    assert s["totals"]["stations"] == 2
