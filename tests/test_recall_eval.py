"""Tests for src/recall_eval.py -- estimating field recall from an exhaustively
annotated sample.

The properties that matter: the plan is reproducible and stratified, empty
segments count as effort rather than being silently dropped, a missed call is
scored as missed, and the interval widens honestly on small samples.
"""

import os
import sys

os.environ.setdefault("PRIMATE_DATA_ROOT", "/tmp/test_data")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd  # noqa: E402
import recall_eval as re_  # noqa: E402


def _recordings(n_per_station=3, stations=("IPA1ST", "IPA2ST"), duration=3600.0):
    return pd.DataFrame([
        {"site": st, "recording": f"{st}_rec{i}", "duration_s": duration}
        for st in stations for i in range(n_per_station)])


# ---------------------------------------------------------------- planning

def test_plan_is_reproducible_from_its_seed():
    a = re_.plan_segments(_recordings(), n_segments=10, seed=42)
    b = re_.plan_segments(_recordings(), n_segments=10, seed=42)
    pd.testing.assert_frame_equal(a, b)
    c = re_.plan_segments(_recordings(), n_segments=10, seed=43)
    assert not a.equals(c)


def test_plan_spreads_segments_across_stations():
    plan = re_.plan_segments(_recordings(stations=("A", "B", "C", "D")),
                             n_segments=12, seed=0)
    counts = plan.groupby("site").size()
    assert len(counts) == 4
    assert counts.min() == counts.max() == 3


def test_remainder_is_spread_not_dumped_on_one_station():
    plan = re_.plan_segments(_recordings(stations=("A", "B", "C")),
                             n_segments=7, seed=0)
    counts = sorted(plan.groupby("site").size().tolist())
    assert counts == [2, 2, 3]


def test_segments_stay_inside_the_recording():
    plan = re_.plan_segments(_recordings(duration=600.0), n_segments=8,
                             segment_s=300.0, seed=1)
    assert (plan["start_s"] >= 0).all()
    assert (plan["end_s"] <= 600.0 + 1e-6).all()


def test_segments_do_not_overlap_within_a_recording():
    plan = re_.plan_segments(
        pd.DataFrame([{"site": "A", "recording": "r", "duration_s": 3600.0}]),
        n_segments=6, segment_s=300.0, seed=3)
    for _, sub in plan.groupby("recording"):
        s = sub.sort_values("start_s")
        assert (s["start_s"].shift(-1).dropna()
                >= s["end_s"].iloc[:-1].values).all()


def test_recordings_shorter_than_a_segment_are_excluded():
    recs = pd.DataFrame([
        {"site": "A", "recording": "short", "duration_s": 60.0},
        {"site": "A", "recording": "long", "duration_s": 3600.0}])
    plan = re_.plan_segments(recs, n_segments=4, segment_s=300.0, seed=0)
    assert set(plan["recording"]) == {"long"}


def test_empty_inputs_do_not_raise():
    assert len(re_.plan_segments(pd.DataFrame())) == 0
    assert len(re_.plan_segments(_recordings(duration=10.0), segment_s=300.0)) == 0


# ------------------------------------------------------------ the template

def test_template_keeps_one_row_per_segment_with_blank_call_times():
    plan = re_.plan_segments(_recordings(), n_segments=4, seed=0)
    t = re_.annotation_template(plan, species="Cernic")
    assert len(t) == len(plan)
    assert (t["call_start_s"] == "").all()
    assert (t["species"] == "Cernic").all()
    assert {"segment_start_s", "segment_end_s"} <= set(t.columns)


# --------------------------------------------------------------- scoring

def _ann(rows):
    """rows: (site, recording, seg_start, seg_end, call_start, call_end)."""
    return pd.DataFrame([{
        "site": s, "recording": r, "segment_start_s": ss, "segment_end_s": se,
        "species": "Cernic", "call_start_s": cs, "call_end_s": cend,
    } for s, r, ss, se, cs, cend in rows])


def _det(rows):
    """rows: (recording, start_s)."""
    return pd.DataFrame([{"recording": r, "start_s": s, "species": "Cernic"}
                         for r, s in rows])


def test_a_detected_call_and_a_missed_call():
    ann = _ann([("A", "r1", 0, 300, 100.0, 103.0),      # detector fires here
                ("A", "r1", 0, 300, 200.0, 203.0)])     # and misses here
    det = _det([("r1", 100.0)])
    r = re_.score_recall(ann, det, window_s=2.0)
    assert r["calls"] == 2 and r["detected"] == 1
    assert r["recall"] == 0.5


def test_overlap_must_exceed_the_minimum_not_merely_touch():
    # Detection [99.8, 101.8) against call [101.5, 104): 0.3 s of overlap.
    ann = _ann([("A", "r1", 0, 300, 101.5, 104.0)])
    det = _det([("r1", 99.8)])
    assert re_.score_recall(ann, det, min_overlap_s=0.5,
                            window_s=2.0)["detected"] == 0
    assert re_.score_recall(ann, det, min_overlap_s=0.2,
                            window_s=2.0)["detected"] == 1


def test_a_detection_in_another_recording_does_not_count():
    ann = _ann([("A", "r1", 0, 300, 100.0, 103.0)])
    det = _det([("r2", 100.0)])
    assert re_.score_recall(ann, det)["detected"] == 0


def test_empty_segments_count_as_listening_effort():
    """A segment with no calls contributes audio time but no calls, and must not
    be dropped -- keeping it is what makes the effort figure honest."""
    ann = _ann([("A", "r1", 0, 300, 100.0, 103.0),
                ("A", "r2", 0, 300, float("nan"), float("nan"))])
    det = _det([("r1", 100.0)])
    r = re_.score_recall(ann, det, window_s=2.0)
    assert r["calls"] == 1 and r["recall"] == 1.0
    assert r["segments"] == 2
    assert r["audio_s"] == 600.0          # both segments were listened to


def test_a_call_with_no_end_time_is_treated_as_one_window():
    ann = _ann([("A", "r1", 0, 300, 100.0, float("nan"))])
    assert re_.score_recall(ann, _det([("r1", 100.0)]),
                            window_s=2.0)["detected"] == 1
    assert re_.score_recall(ann, _det([("r1", 150.0)]),
                            window_s=2.0)["detected"] == 0


def test_per_station_breakdown():
    ann = _ann([("A", "r1", 0, 300, 100.0, 103.0),
                ("B", "r2", 0, 300, 100.0, 103.0),
                ("B", "r2", 0, 300, 200.0, 203.0)])
    det = _det([("r1", 100.0), ("r2", 100.0)])
    per = re_.score_recall(ann, det, window_s=2.0)["per_station"]
    assert per.set_index("site").loc["A", "recall"] == 1.0
    assert per.set_index("site").loc["B", "recall"] == 0.5


def test_no_annotated_calls_returns_none_rather_than_a_fake_recall():
    ann = _ann([("A", "r1", 0, 300, float("nan"), float("nan"))])
    r = re_.score_recall(ann, _det([("r1", 100.0)]))
    assert r["recall"] is None and r["calls"] == 0
    assert r["segments"] == 1


def test_detection_csv_column_names_are_accepted():
    ann = _ann([("A", "r1", 0, 300, 100.0, 103.0)])
    det = pd.DataFrame({"source_file": ["r1.wav"], "start_time": [100.0],
                        "species": ["Cernic"]})
    assert re_.score_recall(ann, det, window_s=2.0)["detected"] == 1


# -------------------------------------------------------------- intervals

def test_wilson_interval_brackets_the_estimate_and_stays_in_range():
    lo, hi = re_.wilson_interval(9, 10)
    assert 0.0 <= lo < 0.9 < hi <= 1.0
    # Unlike the normal approximation, it does not run above 1 at p = 1.
    lo, hi = re_.wilson_interval(10, 10)
    assert hi <= 1.0 and lo < 1.0


def test_interval_narrows_as_the_sample_grows():
    narrow = re_.wilson_interval(90, 100)
    wide = re_.wilson_interval(9, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_wilson_interval_of_nothing_is_none():
    assert re_.wilson_interval(0, 0) == (None, None)


def test_calls_needed_grows_as_the_target_interval_tightens():
    assert (re_.calls_needed_for_ci(0.03) > re_.calls_needed_for_ci(0.05)
            > re_.calls_needed_for_ci(0.10))
    # ~138 calls for +/-5% at an expected recall of 0.9.
    assert 130 <= re_.calls_needed_for_ci(0.05, 0.9) <= 145


def test_segments_needed_translates_calls_into_listening():
    got = re_.segments_needed(calls_needed=100, calls_per_hour=50,
                              segment_s=300.0)
    assert got["audio_hours"] == 2.0
    assert got["segments"] == 24          # 2 h / 5 min
    assert re_.segments_needed(100, 0) is None
