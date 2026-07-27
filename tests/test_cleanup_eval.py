"""Tests for src/cleanup_eval.py -- matching manual-review verdicts against the
automatic cleanup's clean/suspicious split and computing the before/after
numbers reported in the manuscript.
"""

import os
import sys
import tempfile

os.environ.setdefault("PRIMATE_DATA_ROOT", "/tmp/test_data")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd  # noqa: E402
import cleanup_eval  # noqa: E402


def _review_csv(rows, site="IPA1ST", species="Cernic"):
    """rows: list of (clip_filename, manual_id)."""
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "review.csv"), "w") as f:
        f.write('"INDIR","IN FILE","MANUAL ID"\n')
        for fname, mid in rows:
            f.write(f'"/x/{species}/{site}","{fname}","{mid}"\n')
    return d


def _cleanup_dir(clean_rows, susp_rows):
    """rows: list of (species, source_file, start_time, flag_reason)."""
    d = tempfile.mkdtemp()
    cols = ["species", "source_file", "start_time", "flag_reason"]
    pd.DataFrame(clean_rows, columns=cols).to_csv(
        os.path.join(d, "clean_detections.csv"), index=False)
    pd.DataFrame(susp_rows, columns=cols).to_csv(
        os.path.join(d, "suspicious_detections.csv"), index=False)
    return d


def test_headline_numbers():
    rev = _review_csv([
        ("Cernic__recA__00100s__conf0.9.wav", ""),        # call, kept clean
        ("Cernic__recA__00200s__conf0.8.wav", ""),        # call, flagged (lost)
        ("Cernic__recA__00300s__conf0.7.wav", "Noise"),   # FP, flagged (removed)
        ("Cernic__recB__00400s__conf0.6.wav", "Noise"),   # FP, flagged (removed)
        ("Cernic__recB__00500s__conf0.5.wav", "Noise"),   # FP, kept clean (missed)
    ])
    cln = _cleanup_dir(
        clean_rows=[("Cernic", "recA.wav", 100.0, ""),
                    ("Cernic", "recB.wav", 500.0, "")],
        susp_rows=[("Cernic", "recA.wav", 200.0, "isolated"),
                   ("Cernic", "recA.wav", 300.0, "mahalanobis"),
                   ("Cernic", "recB.wav", 400.0, "yamnet:Bird")])

    matched, ev = cleanup_eval.run(rev, cln)
    o = ev["overall"]
    assert o["detections"] == 5
    assert o["calls"] == 2 and o["false_positives"] == 3
    assert o["fp_removed"] == 2 and o["fp_removed_pct"] == 66.7
    assert o["calls_kept"] == 1 and o["calls_lost"] == 1
    assert abs(o["precision_before"] - 0.4) < 1e-9
    assert abs(o["precision_after"] - 0.5) < 1e-9
    assert o["listening_reduction_pct"] == 60.0
    assert ev["unmatched"] == 0
    # only the two removed FPs are attributed, and to the right filters
    assert ev["fp_removed_by_filter"]["mahalanobis"] == 1
    assert ev["fp_removed_by_filter"]["yamnet"] == 1
    assert ev["fp_removed_by_filter"]["temporal isolation"] == 0


def test_start_second_tolerance_and_unmatched():
    rev = _review_csv([
        ("Cernic__recA__00100s__conf0.9.wav", ""),      # matches 100.4 -> 100
        ("Cernic__recA__00201s__conf0.8.wav", "Noise"), # matches 200 within +/-1
        ("Cernic__recA__00900s__conf0.7.wav", "Noise"), # no cleanup row
    ])
    cln = _cleanup_dir(
        clean_rows=[("Cernic", "recA.wav", 100.4, "")],
        susp_rows=[("Cernic", "recA.wav", 200.0, "mahalanobis")])

    matched, ev = cleanup_eval.run(rev, cln, start_tolerance=1)
    assert ev["unmatched"] == 1                 # the 900 s detection
    assert ev["overall"]["detections"] == 2     # unmatched excluded from stats
    assert ev["overall"]["fp_removed"] == 1


def test_species_are_not_cross_matched():
    rev = _review_csv([("Cernic__recA__00100s__conf0.9.wav", "Noise")])
    # same recording and second, but a different species -> must not match
    cln = _cleanup_dir(
        clean_rows=[("Colobus_guereza", "recA.wav", 100.0, "")],
        susp_rows=[])
    matched, ev = cleanup_eval.run(rev, cln)
    assert ev["unmatched"] == 1
    assert ev["overall"]["detections"] == 0


def test_disagreements_are_typed_and_exclude_agreements():
    rev = _review_csv([
        ("Cernic__recA__00100s__conf0.9.wav", ""),        # call, clean -> agree
        ("Cernic__recA__00200s__conf0.8.wav", ""),        # call, flagged -> wrongly_flagged
        ("Cernic__recA__00300s__conf0.7.wav", "Noise"),   # FP, flagged -> agree
        ("Cernic__recB__00400s__conf0.6.wav", "Noise"),   # FP, clean -> missed
    ])
    cln = _cleanup_dir(
        clean_rows=[("Cernic", "recA.wav", 100.0, ""),
                    ("Cernic", "recB.wav", 400.0, "")],
        susp_rows=[("Cernic", "recA.wav", 200.0, "yamnet:Bird"),
                   ("Cernic", "recA.wav", 300.0, "mahalanobis")])

    matched, _ = cleanup_eval.run(rev, cln)
    dis = cleanup_eval.disagreements(matched)

    assert len(dis) == 2                       # the two agreements are excluded
    by_kind = dict(zip(dis["disagreement"], dis["start_s"]))
    assert by_kind[cleanup_eval.WRONGLY_FLAGGED] == 200
    assert by_kind[cleanup_eval.MISSED] == 400
    # the flag that caused the costly error is carried through for diagnosis
    wrong = dis[dis["disagreement"] == cleanup_eval.WRONGLY_FLAGGED].iloc[0]
    assert wrong["flag_reason"] == "yamnet:Bird"


def _cleanup_dir_with_flags(clean_rows, susp_rows):
    """rows: (species, source_file, start_time, mahal, yamnet, isolated)."""
    import tempfile as _t
    d = _t.mkdtemp()
    cols = ["species", "source_file", "start_time",
            "flag_mahal", "flag_yamnet", "flag_isolated"]
    pd.DataFrame(clean_rows, columns=cols).to_csv(
        os.path.join(d, "clean_detections.csv"), index=False)
    pd.DataFrame(susp_rows, columns=cols).to_csv(
        os.path.join(d, "suspicious_detections.csv"), index=False)
    return d


def test_per_filter_lift_separates_useful_from_harmful():
    """A filter that fires mostly on false positives has lift > 1; one that
    fires mostly on genuine calls has lift < 1 and lowers precision."""
    rows, clean, susp = [], [], []
    # 4 genuine calls: yamnet flags 3 of them, mahal flags none
    for i, s in enumerate([100, 200, 300, 400]):
        rows.append((f"Cernic__recA__{s:05d}s__conf0.9.wav", ""))
        yam = i < 3
        entry = ("Cernic", "recA.wav", float(s), False, yam, False)
        (susp if yam else clean).append(entry)
    # 4 false positives: mahal flags 3, yamnet flags 1
    for i, s in enumerate([500, 600, 700, 800]):
        rows.append((f"Cernic__recA__{s:05d}s__conf0.9.wav", "Noise"))
        mah, yam = i < 3, i == 3
        entry = ("Cernic", "recA.wav", float(s), mah, yam, False)
        (susp if (mah or yam) else clean).append(entry)

    matched, _ = cleanup_eval.run(_review_csv(rows),
                                  _cleanup_dir_with_flags(clean, susp))
    pf = cleanup_eval.per_filter_analysis(matched)

    # mahalanobis: 3 of 4 FPs, 0 calls -> strongly useful
    assert pf.loc["mahalanobis", "fps_flagged"] == 3
    assert pf.loc["mahalanobis", "calls_flagged"] == 0
    assert pf.loc["mahalanobis", "precision_if_only"] > 0.5
    # yamnet: 3 of 4 calls, 1 FP -> harmful, lift below 1
    assert pf.loc["yamnet", "calls_flagged"] == 3
    assert pf.loc["yamnet", "lift"] < 1
    assert pf.loc["yamnet", "precision_if_only"] < 0.5


def test_filter_combinations_rank_by_precision():
    """Every subset is derivable from one run's flags, and a subset containing
    a harmful filter must rank below the same subset without it."""
    rows, clean, susp = [], [], []
    for i, s in enumerate([100, 200, 300, 400]):          # genuine calls
        rows.append((f"Cernic__recA__{s:05d}s__conf0.9.wav", ""))
        yam = i < 3                                       # yamnet flags 3 calls
        entry = ("Cernic", "recA.wav", float(s), False, yam, False)
        (susp if yam else clean).append(entry)
    for i, s in enumerate([500, 600, 700, 800]):          # false positives
        rows.append((f"Cernic__recA__{s:05d}s__conf0.9.wav", "Noise"))
        mah = i < 3                                       # mahal flags 3 FPs
        entry = ("Cernic", "recA.wav", float(s), mah, False, False)
        (susp if mah else clean).append(entry)

    matched, _ = cleanup_eval.run(_review_csv(rows),
                                  _cleanup_dir_with_flags(clean, susp))
    fc = cleanup_eval.filter_combination_analysis(matched)

    assert len(fc) == 8                                   # 2^3 subsets
    assert fc.index[0] == "mahalanobis"                   # best subset wins
    # adding the harmful filter must not improve on mahalanobis alone
    assert (fc.loc["mahalanobis + yamnet", "precision"]
            < fc.loc["mahalanobis", "precision"])
    # the no-filter row reproduces the raw precision
    assert fc.loc["(no filter)", "precision"] == 0.5
