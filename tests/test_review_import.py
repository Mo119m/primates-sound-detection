"""Tests for src/review_import.py -- parsing Kaleidoscope review CSVs and
tallying the field-deployment numbers. A tiny synthetic CSV mirrors the real
Wildlife Acoustics column layout.
"""

import os
import sys
import tempfile

os.environ.setdefault("PRIMATE_DATA_ROOT", "/tmp/test_data")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import review_import  # noqa: E402

_HEADER = '"INDIR","IN FILE","MANUAL ID"\n'


def _row(indir, fname, manual):
    return f'"{indir}","{fname}","{manual}"\n'


def _write_csv(rows):
    p = os.path.join(tempfile.mkdtemp(), "IPA1.csv")
    with open(p, "w") as f:
        f.write(_HEADER)
        f.writelines(rows)
    return p


def test_parse_site_species_and_verdict():
    indir = "/Users/x/Downloads/Cernic/IPA1ST"
    rows = [
        _row(indir, "Cernic__20210222T053000+0100_Makokou__01540s__conf0.980.wav", ""),      # call
        _row(indir, "Cernic__20210222T060000+0100_Makokou__01490s__conf0.735.wav", "Noise"),  # FP
        _row(indir, "Cernic__20210222T060000+0100_Makokou__01546s__conf0.786.wav", "noise"),  # FP (case)
    ]
    df = review_import.load_review_csv(_write_csv(rows), blank_is_confirmed=True)
    assert list(df["site"].unique()) == ["IPA1ST"]
    assert list(df["species"].unique()) == ["Cernic"]
    assert list(df["verdict"]) == ["call", "false_positive", "false_positive"]
    assert df.iloc[0]["confidence"] == 0.980
    assert df.iloc[0]["start_s"] == 1540


def test_blank_convention_flips_confirmed():
    indir = "/Users/x/Downloads/Cernic/IPA1ST"
    rows = [
        _row(indir, "Cernic__20210222T053000+0100_Makokou__01540s__conf0.98.wav", ""),
        _row(indir, "Cernic__20210222T060000+0100_Makokou__01490s__conf0.73.wav", "Noise"),
    ]
    p = _write_csv(rows)
    yes = review_import.summarize(review_import.load_review_csv(p, blank_is_confirmed=True))
    no = review_import.summarize(review_import.load_review_csv(p, blank_is_confirmed=False))
    assert yes["totals"]["confirmed"] == 1 and yes["totals"]["unreviewed"] == 0
    assert no["totals"]["confirmed"] == 0 and no["totals"]["unreviewed"] == 1
    # false positives are unaffected by the blank convention
    assert yes["totals"]["false_positive"] == no["totals"]["false_positive"] == 1


def test_precision_and_named_tag_is_confirmed():
    indir = "/Users/x/Downloads/Colobus_guereza/IPA2ST"
    rows = [
        _row(indir, "Colobus_guereza__t__00010s__conf0.9.wav", "Colobus"),   # named tag -> call
        _row(indir, "Colobus_guereza__t__00020s__conf0.8.wav", ""),          # blank -> call
        _row(indir, "Colobus_guereza__t__00030s__conf0.7.wav", "Noise"),     # FP
    ]
    s = review_import.summarize(review_import.load_review_csv(_write_csv(rows)))
    ps = s["per_species"]
    assert int(ps.loc["Colobus_guereza", "confirmed"]) == 2
    assert int(ps.loc["Colobus_guereza", "false_positive"]) == 1
    assert abs(float(ps.loc["Colobus_guereza", "precision"]) - (2 / 3)) < 1e-4
