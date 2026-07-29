"""Tests for src/review_queue.py -- the label-free review queue the pipeline
emits: episodes grouped, clips ordered most-likely-genuine first.

The properties that matter are that it needs no labels, that it drops nothing,
that its ordering matches the one the paper scores, and that one row per episode
is marked as the clip to listen to first.
"""

import os
import sys
import tempfile

os.environ.setdefault("PRIMATE_DATA_ROOT", "/tmp/test_data")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd  # noqa: E402
import cleanup_eval  # noqa: E402
import review_queue as rq  # noqa: E402


def _det(rows, site="IPA1ST", species="Cernic"):
    """rows: list of (recording, start_s, confidence, margin, neighbours, mahal)."""
    return pd.DataFrame([{
        "site": site, "species": species, "recording": rec, "start_s": float(s),
        "confidence": c, "softmax_margin": m, "n_neighbours": n,
        "mahalanobis_d2": d,
    } for rec, s, c, m, n, d in rows])


def test_needs_no_labels_and_keeps_every_detection():
    df = _det([("recA", 0, 0.9, 0.8, 5, 100.0),
               ("recA", 10, 0.5, 0.4, 5, 900.0),
               ("recB", 5000, 0.7, 0.6, 0, 500.0)])
    assert "verdict" not in df.columns
    q = rq.build(df)
    assert len(q) == 3
    assert q["rank_score"].notna().all()


def test_ordering_matches_the_evaluation_module():
    """The queue a reviewer works from must be the ordering the paper scores.
    Two implementations of the same rank average have to agree."""
    df = _det([("recA", 0, 0.95, 0.90, 6, 120.0),
               ("recA", 30, 0.60, 0.50, 2, 800.0),
               ("recA", 60, 0.80, 0.70, 4, 400.0),
               ("recB", 9000, 0.70, 0.65, 1, 650.0)])
    mine = rq.rank_score(df).round(6)

    # cleanup_eval works on the matched table, which carries a 'cleanup' column.
    theirs = cleanup_eval.review_ranking(df.assign(cleanup="clean")).round(6)
    assert mine.sort_index().equals(theirs.sort_index())


def test_one_listen_first_row_per_episode():
    df = _det([("recA", 0, 0.9, 0.8, 5, 100.0),
               ("recA", 10, 0.5, 0.4, 5, 900.0),
               ("recA", 20, 0.6, 0.5, 5, 800.0),
               ("recB", 9000, 0.7, 0.6, 0, 500.0)])
    q = rq.build(df)
    assert int(q["listen_first"].sum()) == q["episode"].nunique() == 2
    # The marked row is its episode's best-ranked clip, since that is the clip
    # the reviewer actually hears before judging the episode.
    for _, sub in q.groupby("episode"):
        best = sub["rank_score"].max()
        assert sub.loc[sub["listen_first"], "rank_score"].iloc[0] == best


def test_episodes_are_emitted_best_first_and_stay_contiguous():
    good = _det([("recA", t, 0.99, 0.98, 8, 50.0) for t in (0, 10, 20)])
    bad = _det([("recB", t, 0.30, 0.20, 1, 990.0) for t in (9000, 9010, 9020)])
    q = rq.build(pd.concat([good, bad], ignore_index=True))

    assert q["recording"].iloc[0] == "recA"          # best episode leads
    # An episode is not interleaved with another; a reviewer works it as a block.
    assert q["episode"].tolist() == sorted(q["episode"].tolist(),
                                          key=lambda e: q.index[q["episode"] == e][0])
    runs = (q["episode"] != q["episode"].shift()).sum()
    assert runs == q["episode"].nunique()


def test_ranks_are_computed_within_station_not_across():
    """Station B's confidences are all lower than station A's. If the rank were
    pooled, every B detection would sort below every A detection; within-station
    ranking has to put B's best above A's worst."""
    a = _det([("recA", 0, 0.99, 0.98, 5, 100.0),
              ("recA", 30, 0.90, 0.85, 5, 100.0)], site="IPA1ST")
    b = _det([("recB", 0, 0.50, 0.45, 5, 100.0),
              ("recB", 30, 0.40, 0.35, 5, 100.0)], site="IPA2ST")
    s = rq.rank_score(pd.concat([a, b], ignore_index=True))
    # Best of station B (index 2) outranks worst of station A (index 1).
    assert s.iloc[2] > s.iloc[1]


def test_falls_back_to_chronological_without_signals_rather_than_faking_one():
    df = pd.DataFrame({"site": ["IPA1ST"] * 3, "species": ["Cernic"] * 3,
                       "recording": ["recA"] * 3, "start_s": [20.0, 0.0, 10.0]})
    q = rq.build(df)
    assert len(q) == 3
    assert q["rank_score"].isna().all()          # no ordering claimed
    assert q["start_s"].tolist() == [0.0, 10.0, 20.0]


def test_accepts_detection_csv_column_names():
    df = pd.DataFrame({"site": ["IPA1ST"], "species": ["Cernic"],
                       "source_file": ["recA.wav"], "start_time": [12.0],
                       "confidence": [0.9], "softmax_margin": [0.8],
                       "n_neighbours": [3], "mahalanobis_d2": [100.0]})
    q = rq.build(df)
    assert q["recording"].iloc[0] == "recA"
    assert q["start_s"].iloc[0] == 12.0


def test_kaleidoscope_layout_and_blank_manual_id():
    df = _det([("recA", 12, 0.90, 0.8, 5, 100.0)])
    k = rq.to_kaleidoscope(rq.build(df), clips_root="/clips")
    assert list(k.columns[:3]) == ["INDIR", "IN FILE", "MANUAL ID"]
    assert k["MANUAL ID"].iloc[0] == ""
    assert k["INDIR"].iloc[0] == os.path.join("/clips", "Cernic", "IPA1ST")
    # Matches the name extract_clips() writes: <recording>__t<start>s__conf<c>.wav
    assert k["IN FILE"].iloc[0] == "recA__t00012s__conf0.90.wav"


def test_existing_clip_filenames_are_used_when_present():
    df = _det([("recA", 12, 0.9, 0.8, 5, 100.0)])
    df["clip_file"] = ["whatever_santi_called_it.wav"]
    k = rq.to_kaleidoscope(rq.build(df))
    assert k["IN FILE"].iloc[0] == "whatever_santi_called_it.wav"


def test_save_writes_both_files_and_the_episode_summary():
    df = _det([("recA", 0, 0.9, 0.8, 5, 100.0),
               ("recA", 10, 0.5, 0.4, 5, 900.0),
               ("recB", 9000, 0.7, 0.6, 0, 500.0)])
    d = tempfile.mkdtemp()
    q = rq.save(df, d, verbose=False)

    queue = pd.read_csv(os.path.join(d, rq.QUEUE_FILENAME))
    assert len(queue) == 3
    eps = pd.read_csv(os.path.join(d, "review_queue_episodes.csv"))
    assert len(eps) == q["episode"].nunique() == 2
    # The summary says how many detections each single listen settles.
    assert eps["episode_detections"].sum() == 3


def test_empty_input_does_not_raise():
    empty = pd.DataFrame(columns=["site", "species", "recording", "start_s"])
    assert len(rq.build(empty)) == 0
    assert list(rq.to_kaleidoscope(rq.build(empty)).columns) == [
        "INDIR", "IN FILE", "MANUAL ID"]
