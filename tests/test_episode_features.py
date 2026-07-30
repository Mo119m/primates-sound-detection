"""Tests for src/episode_features.py -- temporal structure derived from a
detection table with no labels, no audio and no model."""

import os
import sys

os.environ.setdefault("PRIMATE_DATA_ROOT", "/tmp/test_data")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import episode_features as ef  # noqa: E402


def _det(rows, site="IPA1ST", species="Cernic"):
    """rows: list of (recording, start_s)."""
    return pd.DataFrame([{"site": site, "species": species,
                          "recording": rec, "start_s": float(s)}
                         for rec, s in rows])


def test_gap_splits_episodes():
    # Two detections 10 s apart, then one 20 min later.
    df = _det([("recA", 0), ("recA", 10), ("recA", 1200)])
    out = ef.assign_episodes(df, gap_s=300.0)
    assert out["episode"].tolist() == [1, 1, 2]


def test_recording_and_species_always_split():
    df = pd.concat([
        _det([("recA", 100)]),
        _det([("recB", 101)]),                      # different recording
        _det([("recA", 102)], species="Colobus"),   # different species
        _det([("recA", 103)], site="IPA2ST"),       # different station
    ], ignore_index=True)
    out = ef.assign_episodes(df, gap_s=300.0)
    assert out["episode"].nunique() == 4


def test_row_order_is_preserved():
    """The grouping sorts internally; callers must get their own order back."""
    df = _det([("recA", 1200), ("recA", 0), ("recA", 10)])
    out = ef.assign_episodes(df, gap_s=300.0)
    # Row 0 is the late detection, so it is the *second* episode.
    assert out["episode"].tolist() == [2, 1, 1]
    assert out["start_s"].tolist() == [1200.0, 0.0, 10.0]


def test_dense_run_scores_a_higher_rate_than_a_sparse_bout():
    """The feature that is supposed to separate a running intruder from a
    primate bout has to actually do so on the clearest possible case."""
    dense = _det([("recA", t) for t in range(0, 100, 2)])        # 50 in 98 s
    sparse = _det([("recB", t) for t in (0, 40, 80, 120, 160)])  # 5 in 160 s
    out = ef.add_episode_features(pd.concat([dense, sparse], ignore_index=True))

    rate_dense = out.loc[out["recording"] == "recA", "episode_rate"].iloc[0]
    rate_sparse = out.loc[out["recording"] == "recB", "episode_rate"].iloc[0]
    assert rate_dense > rate_sparse

    gap_dense = out.loc[out["recording"] == "recA", "episode_mean_gap_s"].iloc[0]
    gap_sparse = out.loc[out["recording"] == "recB", "episode_mean_gap_s"].iloc[0]
    assert gap_dense == 2.0 and gap_sparse == 40.0


def test_singleton_gets_the_grouping_gap_not_an_infinity():
    df = _det([("recA", 0)])
    out = ef.add_episode_features(df, gap_s=300.0)
    assert out["episode_size"].iloc[0] == 1.0
    assert out["episode_span_s"].iloc[0] == 0.0
    assert out["episode_mean_gap_s"].iloc[0] == 300.0
    assert np.isfinite(out["episode_rate"].iloc[0])
    assert out["episode_position"].iloc[0] == 0.0


def test_episode_position_runs_from_zero_to_one():
    df = _det([("recA", 0), ("recA", 50), ("recA", 100)])
    out = ef.add_episode_features(df)
    assert out["episode_position"].tolist() == [0.0, 0.5, 1.0]


def test_gap_to_nearest_is_the_smaller_of_the_two_neighbours():
    df = _det([("recA", 0), ("recA", 10), ("recA", 100)])
    g = ef.add_episode_features(df)["gap_to_nearest_s"]
    assert g.tolist() == [10.0, 10.0, 90.0]


def test_gap_to_nearest_is_nan_for_a_lone_detection():
    df = _det([("recA", 0)])
    assert pd.isna(ef.add_episode_features(df)["gap_to_nearest_s"].iloc[0])


def test_gap_to_nearest_does_not_cross_recordings():
    """A detection 1 s away in a *different* recording is not a neighbour."""
    df = _det([("recA", 100), ("recB", 101)])
    g = ef.add_episode_features(df)["gap_to_nearest_s"]
    assert g.isna().all()


def test_empty_and_missing_columns_do_not_raise():
    empty = pd.DataFrame(columns=["site", "species", "recording", "start_s"])
    assert "episode" in ef.assign_episodes(empty).columns

    no_time = pd.DataFrame({"site": ["IPA1ST"], "species": ["Cernic"]})
    out = ef.add_episode_features(no_time)
    assert "episode_rate" not in out.columns  # absent, not guessed


def test_available_candidates_reports_what_was_computed():
    # Episodes of differing size and spacing, so every feature actually varies;
    # a fixture where they are all constant would exercise nothing.
    out = ef.add_episode_features(_det(
        [("recA", 0), ("recA", 10), ("recA", 40),      # 3 detections, 40 s
         ("recA", 900), ("recA", 902),                 # 2 detections, 2 s
         ("recA", 5000)]))                             # singleton
    got = ef.available_candidates(out, include_events=False)
    assert set(got) == set(ef.CANDIDATE_SIGNALS)
    # gap_to_nearest_s is all-NaN when nothing has a neighbour, and is then
    # correctly reported as unavailable rather than as a column of NaN.
    lone = ef.add_episode_features(_det([("recA", 0)]))
    assert "gap_to_nearest_s" not in ef.available_candidates(lone)


# ------------------------------------------------- call-event scale features

def test_event_features_measure_the_vocalization_not_the_bout():
    """A 3-window call and a lone window are one episode but two events."""
    df = _det([("recA", 0), ("recA", 1), ("recA", 2), ("recA", 60)])
    out = ef.add_event_features(df, window_s=2.0)
    assert out["event"].nunique() == 2
    assert out["event_windows"].tolist() == [3.0, 3.0, 3.0, 1.0]
    assert out["event_duration_s"].tolist() == [4.0, 4.0, 4.0, 2.0]
    # ... while the episode grouping keeps all four together.
    assert ef.assign_episodes(df)["episode"].nunique() == 1


def test_event_position_runs_zero_to_one_within_the_call():
    out = ef.add_event_features(_det([("recA", 0), ("recA", 1), ("recA", 2)]))
    assert out["event_position"].tolist() == [0.0, 0.5, 1.0]


def test_add_all_features_gives_both_scales():
    out = ef.add_all_features(_det([("recA", 0), ("recA", 1), ("recA", 600)]))
    assert {"episode", "event"} <= set(out.columns)
    assert set(ef.EVENT_CANDIDATE_SIGNALS) <= set(out.columns)
    assert set(ef.CANDIDATE_SIGNALS) <= set(out.columns)


def test_available_candidates_can_include_or_exclude_the_event_scale():
    out = ef.add_all_features(_det([("recA", 0), ("recA", 1), ("recA", 600)]))
    with_ev = ef.available_candidates(out, include_events=True)
    without = ef.available_candidates(out, include_events=False)
    assert "event_windows" in with_ev and "event_windows" not in without


def test_a_constant_feature_is_not_offered_as_a_candidate():
    """Every detection isolated -> event_windows is 1 everywhere. A column with
    one value carries no ordering, so offering it would only add tied ranks."""
    out = ef.add_all_features(_det([("recA", 0), ("recA", 600), ("recA", 1200)]))
    assert out["event_windows"].nunique() == 1
    assert "event_windows" not in ef.available_candidates(out)
