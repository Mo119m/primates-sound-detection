"""Tests for src/field_gate.py -- deciding whether a retrained model earns a
full detection run.

The gate exists to stop two specific mistakes, so those are what the tests are
built around: buying precision by throwing away genuine calls, and improving
only at the stations the hard negatives were mined from.
"""

import os
import sys

os.environ.setdefault("PRIMATE_DATA_ROOT", "/tmp/test_data")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import field_gate as fg  # noqa: E402


def _reviewed(spec):
    """spec: {station: (n_calls, n_false_positives)}."""
    rows, t = [], 0.0
    for st, (n_calls, n_fps) in spec.items():
        for i in range(n_calls):
            rows.append({"site": st, "species": "Cernic", "recording": f"{st}_r",
                         "start_s": t, "verdict": "call", "confidence": 0.9})
            t += 10.0
        for i in range(n_fps):
            rows.append({"site": st, "species": "Cernic", "recording": f"{st}_r",
                         "start_s": t, "verdict": "false_positive",
                         "confidence": 0.8})
            t += 10.0
    return pd.DataFrame(rows)


def _predictions(reviewed, keep_call=1.0, keep_fp=1.0, stations=None,
                 conf=0.99):
    """Emit predictions that keep the given share of calls / false positives.

    ``stations`` restricts the *removal* to those stations, which is how a
    mined-station-only improvement is simulated.
    """
    rows = []
    seen = {}
    for r in reviewed.itertuples():
        share = keep_call if r.verdict == "call" else keep_fp
        if stations is not None and str(r.site) not in stations:
            share = 1.0
        k = (r.site, r.verdict)
        seen[k] = seen.get(k, 0) + 1
        n_of_kind = ((reviewed["site"] == r.site)
                     & (reviewed["verdict"] == r.verdict)).sum()
        keep = seen[k] <= round(share * n_of_kind)
        rows.append({"recording": r.recording, "start_s": r.start_s,
                     "pred_species": "Cernic" if keep else "Background",
                     "pred_confidence": conf})
    return pd.DataFrame(rows)


def _scored(reviewed, **kw):
    pred = _predictions(reviewed, **kw)
    s, _ = fg.apply_candidate(reviewed, pred, species="Cernic")
    return s


# ------------------------------------------------------------- joining

def test_join_matches_on_recording_and_rounded_time():
    rev = _reviewed({"A": (2, 2)})
    pred = _predictions(rev)
    pred["start_s"] = pred["start_s"] + 0.2      # sub-second write difference
    scored, missing = fg.apply_candidate(rev, pred, species="Cernic")
    assert missing == 0 and len(scored) == len(rev)


def test_clips_with_no_prediction_are_reported_not_assumed():
    rev = _reviewed({"A": (2, 2)})
    pred = _predictions(rev).iloc[:2]
    scored, missing = fg.apply_candidate(rev, pred, species="Cernic")
    assert missing == 2 and len(scored) == 2


def test_a_confidence_threshold_can_drop_a_target_prediction():
    rev = _reviewed({"A": (1, 0)})
    pred = _predictions(rev, conf=0.30)
    kept_hi, _ = fg.apply_candidate(rev, pred, species="Cernic", threshold=0.5)
    kept_lo, _ = fg.apply_candidate(rev, pred, species="Cernic", threshold=0.2)
    assert not kept_hi["kept"].any()
    assert kept_lo["kept"].all()


def test_missing_join_keys_raise_rather_than_silently_score_nothing():
    rev = _reviewed({"A": (1, 1)}).drop(columns=["recording", "start_s"])
    with pytest.raises(ValueError):
        fg.apply_candidate(rev, _predictions(_reviewed({"A": (1, 1)})))


# ------------------------------------------------------------- scoring

def test_a_perfect_candidate_removes_every_false_positive():
    s = fg.score(_scored(_reviewed({"A": (10, 90)}), keep_fp=0.0))
    assert s["precision_before"] == 0.1
    assert s["precision_after"] == 1.0
    assert s["call_retention"] == 1.0
    assert s["fp_removal"] == 1.0


def test_a_do_nothing_candidate_changes_nothing():
    s = fg.score(_scored(_reviewed({"A": (10, 90)})))
    assert s["precision_gain"] == 0.0
    assert s["fps_removed"] == 0


def test_losing_calls_and_false_positives_equally_does_not_help():
    s = fg.score(_scored(_reviewed({"A": (10, 90)}), keep_call=0.5, keep_fp=0.5))
    assert abs(s["precision_gain"]) < 0.01


# --------------------------------------------------------------- gating

def test_rejects_a_candidate_that_buys_precision_by_losing_calls():
    """Removes 90% of false positives, but half the genuine calls with them."""
    g = fg.gate(_scored(_reviewed({"A": (100, 900), "B": (100, 900)}),
                        keep_call=0.5, keep_fp=0.1))
    assert g["verdict"] == "reject"
    assert "genuine calls" in g["reason"]


def test_rejects_a_candidate_that_does_nothing():
    g = fg.gate(_scored(_reviewed({"A": (100, 900), "B": (100, 900)})))
    assert g["verdict"] == "reject"
    assert "below the" in g["reason"]


def test_passes_a_candidate_to_pilot_never_straight_to_ship():
    """A big honest gain still only earns a pilot run: the reviewed clips cannot
    show false positives the old model never reported."""
    g = fg.gate(_scored(_reviewed({"A": (100, 900), "B": (100, 900)}),
                        keep_call=0.99, keep_fp=0.1))
    assert g["verdict"] == "pilot"
    assert "continuous audio" in g["reason"]


def test_catches_an_improvement_confined_to_the_mined_station():
    """The failure this whole evaluation exists to catch. IPA4ST's noise is
    removed; the other stations are untouched."""
    rev = _reviewed({"IPA4ST": (100, 2000), "B": (100, 200), "C": (100, 200)})
    pred = _predictions(rev, keep_fp=0.05, stations={"IPA4ST"})
    scored, _ = fg.apply_candidate(rev, pred, species="Cernic")

    # Pooled, this looks like a triumph: 11.1% -> 37.5%, because the mined
    # station holds most of the false positives in the whole set.
    assert fg.score(scored)["precision_gain"] > 0.25
    # Gated against the unmined stations, it is not one.
    g = fg.gate(scored, mined_from=["IPA4ST"])
    assert g["verdict"] == "mined-only"
    assert g["mined"]["precision_gain"] > 0.30
    assert abs(g["unmined"]["precision_gain"]) < 0.01


def test_a_gain_that_holds_off_the_mined_station_passes():
    rev = _reviewed({"IPA4ST": (100, 2000), "B": (100, 200), "C": (100, 200)})
    pred = _predictions(rev, keep_call=0.99, keep_fp=0.1)
    scored, _ = fg.apply_candidate(rev, pred, species="Cernic")
    g = fg.gate(scored, mined_from=["IPA4ST"])
    assert g["verdict"] == "pilot"
    assert g["unmined"]["precision_gain"] > 0.05


def test_the_unmined_score_is_what_the_verdict_uses():
    """Even a huge pooled gain must not override a flat unmined result."""
    rev = _reviewed({"IPA4ST": (10, 5000), "B": (100, 100)})
    pred = _predictions(rev, keep_fp=0.0, stations={"IPA4ST"})
    scored, _ = fg.apply_candidate(rev, pred, species="Cernic")
    g = fg.gate(scored, mined_from=["IPA4ST"])
    assert g["overall"]["precision_gain"] > 0.5
    assert g["verdict"] == "mined-only"


# ------------------------------------------------------------ reporting

def test_per_station_breaks_the_result_out():
    rev = _reviewed({"A": (10, 90), "B": (10, 90)})
    pred = _predictions(rev, keep_fp=0.0, stations={"A"})
    scored, _ = fg.apply_candidate(rev, pred, species="Cernic")
    ps = fg.per_station(scored).set_index("site")
    assert ps.loc["A", "precision_after"] == 1.0
    assert ps.loc["B", "precision_after"] == 0.1


def test_lost_calls_lists_what_the_candidate_would_drop():
    rev = _reviewed({"A": (10, 90)})
    scored = _scored(rev, keep_call=0.8)
    lost = fg.lost_calls(scored)
    assert len(lost) == 2
    assert (lost["verdict"] == "call").all() if "verdict" in lost else True


def test_compare_puts_two_candidates_side_by_side():
    rev = _reviewed({"A": (10, 90)})
    out = fg.compare(_scored(rev), _scored(rev, keep_fp=0.0))
    assert list(out.index) == ["current", "candidate"]
    assert out.loc["candidate", "precision_after"] > out.loc["current", "precision_after"]


def test_summary_text_leads_with_the_unmined_line():
    rev = _reviewed({"IPA4ST": (100, 2000), "B": (100, 200)})
    pred = _predictions(rev, keep_fp=0.05, stations={"IPA4ST"})
    scored, _ = fg.apply_candidate(rev, pred, species="Cernic")
    txt = fg.summarise_text(fg.gate(scored, mined_from=["IPA4ST"]))
    assert "ELSEWHERE" in txt
    assert "decides it" in txt
    assert "MINED-ONLY" in txt


def test_empty_input_is_a_rejection_not_a_crash():
    g = fg.gate(pd.DataFrame(columns=["site", "verdict", "kept"]))
    assert g["verdict"] == "reject"


# ------------------------------------------------- the command-line wrapper

def test_gate_retrain_runs_and_exits_nonzero_on_a_bad_candidate(tmp_path):
    """The exit code is the machine-readable verdict: 0 only for 'pilot', so the
    command can gate a script without parsing its output."""
    import subprocess

    rev = _reviewed({"IPA4ST": (100, 2000), "B": (100, 200)})
    rev_p = tmp_path / "cleanup_vs_review.csv"
    rev.to_csv(rev_p, index=False)

    # A candidate that only fixes the mined station.
    pred_p = tmp_path / "preds.csv"
    _predictions(rev, keep_fp=0.05, stations={"IPA4ST"}).to_csv(pred_p, index=False)

    script = os.path.join(os.path.dirname(__file__), "..", "scripts",
                          "gate_retrain.py")
    proc = subprocess.run(
        [sys.executable, script, "--reviewed", str(rev_p),
         "--predictions", str(pred_p), "--mined-from", "IPA4ST",
         "--out", str(tmp_path)],
        capture_output=True, text=True)
    assert proc.returncode == 1, proc.stdout
    assert "MINED-ONLY" in proc.stdout
    assert (tmp_path / "gate_scored.csv").exists()
    assert (tmp_path / "gate_per_station.csv").exists()


def test_gate_retrain_exits_zero_when_the_candidate_earns_a_pilot(tmp_path):
    import subprocess

    rev = _reviewed({"IPA4ST": (100, 2000), "B": (100, 200), "C": (100, 200)})
    rev_p = tmp_path / "cleanup_vs_review.csv"
    rev.to_csv(rev_p, index=False)
    pred_p = tmp_path / "preds.csv"
    _predictions(rev, keep_call=0.99, keep_fp=0.1).to_csv(pred_p, index=False)

    script = os.path.join(os.path.dirname(__file__), "..", "scripts",
                          "gate_retrain.py")
    proc = subprocess.run(
        [sys.executable, script, "--reviewed", str(rev_p),
         "--predictions", str(pred_p), "--mined-from", "IPA4ST",
         "--out", str(tmp_path)],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout
    assert "PILOT" in proc.stdout
    assert "Stage 3" in proc.stdout        # it says what to do next
