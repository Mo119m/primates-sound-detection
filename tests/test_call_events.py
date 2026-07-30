"""Tests for src/call_events.py -- consolidating overlapping detection windows
into call events.

The point of the module is that a detection count is a count of windows, not of
vocalizations, because a 2 s window sliding 1 s at a time sees a long call
several times and per-species NMS keeps those neighbours on purpose. These tests
pin down the merging rule, the arithmetic of duration, and the inflation factor
the module exists to report.
"""

import os
import sys

os.environ.setdefault("PRIMATE_DATA_ROOT", "/tmp/test_data")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd  # noqa: E402
import call_events as ce  # noqa: E402


def _det(rows, site="IPA1ST", species="Cernic", verdict=None):
    """rows: list of (recording, start_s) or (recording, start_s, verdict)."""
    out = []
    for r in rows:
        rec, start = r[0], r[1]
        d = {"site": site, "species": species, "recording": rec,
             "start_s": float(start), "confidence": 0.9}
        v = r[2] if len(r) > 2 else verdict
        if v is not None:
            d["verdict"] = v
        out.append(d)
    return pd.DataFrame(out)


def test_consecutive_overlapping_windows_are_one_event():
    """A 4 s call seen by windows at 0, 1 and 2 s is one call, not three."""
    df = _det([("recA", 0), ("recA", 1), ("recA", 2)])
    assert ce.assign_events(df, max_gap_s=2.0)["event"].nunique() == 1


def test_a_gap_wider_than_the_window_splits_the_event():
    df = _det([("recA", 0), ("recA", 1), ("recA", 10)])
    assert ce.assign_events(df, max_gap_s=2.0)["event"].tolist() == [1, 1, 2]


def test_touching_windows_merge_but_disjoint_ones_do_not():
    # start times 2.0 apart -> windows [0,2] and [2,4] touch exactly: one event.
    assert ce.assign_events(_det([("recA", 0), ("recA", 2)]),
                            max_gap_s=2.0)["event"].nunique() == 1
    # 2.5 apart -> a half-second of silence between them: two events.
    assert ce.assign_events(_det([("recA", 0), ("recA", 2.5)]),
                            max_gap_s=2.0)["event"].nunique() == 2


def test_species_recording_and_station_always_separate_events():
    df = pd.concat([
        _det([("recA", 100)]),
        _det([("recB", 100)]),
        _det([("recA", 100)], species="Colobus_guereza"),
        _det([("recA", 100)], site="IPA2ST"),
    ], ignore_index=True)
    assert ce.assign_events(df)["event"].nunique() == 4


def test_row_order_is_preserved():
    df = _det([("recA", 10), ("recA", 0), ("recA", 1)])
    out = ce.assign_events(df, max_gap_s=2.0)
    assert out["event"].tolist() == [2, 1, 1]
    assert out["start_s"].tolist() == [10.0, 0.0, 1.0]


def test_duration_counts_the_last_window_not_just_the_starts():
    """Windows at 0 and 1 span [0,3), not [0,1]: the event is 3 s long."""
    ev = ce.events(_det([("recA", 0), ("recA", 1)]), max_gap_s=2.0, window_s=2.0)
    assert ev["start_s"].iloc[0] == 0.0
    assert ev["end_s"].iloc[0] == 3.0
    assert ev["duration_s"].iloc[0] == 3.0


def test_a_single_window_event_has_the_window_duration():
    ev = ce.events(_det([("recA", 0)]), window_s=2.0)
    assert ev["duration_s"].iloc[0] == 2.0
    assert ev["windows"].iloc[0] == 1


def test_an_event_is_a_call_when_any_of_its_windows_was_confirmed():
    """The windows are the same sound, so one confirmation settles the event."""
    df = _det([("recA", 0, "false_positive"), ("recA", 1, "call"),
               ("recA", 2, "false_positive")])
    ev = ce.events(df, max_gap_s=2.0)
    assert len(ev) == 1
    assert ev["verdict"].iloc[0] == "call"
    assert ev["call_windows"].iloc[0] == 1
    assert bool(ev["mixed"].iloc[0]) is True


def test_an_all_false_positive_event_stays_a_false_positive():
    df = _det([("recA", 0), ("recA", 1)], verdict="false_positive")
    ev = ce.events(df, max_gap_s=2.0)
    assert ev["verdict"].iloc[0] == "false_positive"
    assert bool(ev["mixed"].iloc[0]) is False


def test_inflation_is_call_windows_over_call_events():
    """The number the module exists to report: how much a window count
    overstates the number of vocalizations."""
    # Two calls, one seen by 3 windows and one by 1 -> 4 windows, 2 events.
    df = _det([("recA", 0), ("recA", 1), ("recA", 2), ("recA", 60)],
              verdict="call")
    eff = ce.event_effort(df, max_gap_s=2.0)
    row = eff.loc["all stations"]
    assert row["call_windows"] == 4 and row["call_events"] == 2
    assert row["inflation"] == 2.0
    assert row["windows"] == 4 and row["events"] == 2


def test_event_effort_reports_per_station_when_there_are_several():
    a = _det([("recA", 0), ("recA", 1)], site="IPA1ST", verdict="call")
    b = _det([("recB", 0)], site="IPA2ST", verdict="false_positive")
    eff = ce.event_effort(pd.concat([a, b], ignore_index=True), max_gap_s=2.0)
    assert set(eff.index) == {"all stations", "IPA1ST", "IPA2ST"}
    assert eff.loc["IPA1ST", "events"] == 1
    assert eff.loc["all stations", "windows"] == 3


def test_duration_summary_splits_by_verdict():
    calls = _det([("recA", 0), ("recA", 1), ("recA", 2)], verdict="call")
    fps = _det([("recA", 100), ("recA", 200)], verdict="false_positive")
    ds = ce.duration_summary(pd.concat([calls, fps], ignore_index=True),
                             max_gap_s=2.0, window_s=2.0)
    assert {"all events", "call", "false_positive"} <= set(ds.index)
    # The call spans three windows; each false positive is its own single window.
    assert ds.loc["call", "mean_windows"] == 3.0
    assert ds.loc["false_positive", "mean_windows"] == 1.0
    assert ds.loc["false_positive", "single_window_pct"] == 100.0


def test_works_without_verdicts_at_all():
    """The merging is by timestamp, so it must run before any review exists."""
    df = _det([("recA", 0), ("recA", 1)])
    assert "verdict" not in df.columns
    ev = ce.events(df, max_gap_s=2.0)
    assert len(ev) == 1 and "verdict" not in ev.columns
    assert len(ce.event_effort(df, max_gap_s=2.0)) == 1


def test_empty_and_timeless_tables_do_not_raise():
    empty = pd.DataFrame(columns=["site", "species", "recording", "start_s"])
    assert "event" in ce.assign_events(empty).columns
    assert len(ce.events(empty)) == 0
    assert len(ce.event_effort(empty)) == 0

    no_time = pd.DataFrame({"site": ["IPA1ST"], "species": ["Cernic"]})
    assert len(ce.events(no_time)) == 0


def test_default_gap_is_the_configured_window_length():
    """The merging condition is 'the windows overlap or touch', which for a
    forward-sliding window is exactly a start-time gap of the window length."""
    import config
    assert ce.DEFAULT_MAX_GAP_S == float(config.WINDOW_SIZE)
