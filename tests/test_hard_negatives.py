"""Tests for src/hard_negatives.py -- choosing which reviewed detections to
fold back into training.

The selection is what decides whether retraining generalises, and the way it
fails is specific: one station supplied 2370 of 3654 confirmed false positives
here, so proportional mining teaches that station's intruder and little else.
These tests pin the balancing, the holdout, and the within-station spread.
"""

import os
import sys

os.environ.setdefault("PRIMATE_DATA_ROOT", "/tmp/test_data")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import hard_negatives as hn  # noqa: E402


def _matched(spec, episodes_per_station=10):
    """spec: {station: (n_calls, n_false_positives)}."""
    rows = []
    for st, (n_calls, n_fps) in spec.items():
        for i in range(n_calls):
            rows.append({"site": st, "species": "Cernic", "recording": f"{st}_r",
                         "start_s": float(i * 600), "verdict": "call",
                         "confidence": 0.95})
        for i in range(n_fps):
            # Spread false positives over several episodes, 10 min apart.
            ep = i % episodes_per_station
            rows.append({"site": st, "species": "Cernic", "recording": f"{st}_r",
                         "start_s": float(100000 + ep * 600 + (i // episodes_per_station)),
                         "verdict": "false_positive", "confidence": 0.85})
    return pd.DataFrame(rows)


# ------------------------------------------------------------ water filling

def test_equal_supply_splits_equally():
    got = hn.water_fill({"A": 100, "B": 100, "C": 100}, 60)
    assert got == {"A": 20, "B": 20, "C": 20}


def test_a_small_group_gives_everything_and_the_rest_absorb_the_shortfall():
    """B can only supply 5 of its 20-item share; A and C take the other 15."""
    got = hn.water_fill({"A": 100, "B": 5, "C": 100}, 60)
    assert got["B"] == 5
    assert sum(got.values()) == 60
    assert got["A"] == got["C"] == 27 or abs(got["A"] - got["C"]) <= 1


def test_the_dominant_station_does_not_take_its_proportional_share():
    """The case this exists for: one station with 2370 false positives and
    fifteen with ~85 each. Proportional mining would give it 65% of the set."""
    avail = {"IPA4ST": 2370}
    avail.update({f"S{i}": 85 for i in range(15)})
    got = hn.water_fill(avail, 1500)
    assert sum(got.values()) == 1500
    # Proportionally it would be ~975; balanced it is a fraction of that.
    assert got["IPA4ST"] < 300
    # ... and the small stations are exhausted rather than crowded out.
    assert all(got[f"S{i}"] == 85 for i in range(15))


def test_asking_for_more_than_exists_returns_everything():
    got = hn.water_fill({"A": 10, "B": 20}, 1000)
    assert got == {"A": 10, "B": 20}


def test_fewer_items_than_groups_is_deterministic():
    got = hn.water_fill({"A": 100, "B": 50, "C": 10}, 2)
    assert sum(got.values()) == 2
    assert hn.water_fill({"A": 100, "B": 50, "C": 10}, 2) == got


def test_zero_and_empty_are_handled():
    assert sum(hn.water_fill({"A": 10}, 0).values()) == 0
    assert hn.water_fill({}, 100) == {}
    assert hn.water_fill({"A": 0, "B": 10}, 5) == {"A": 0, "B": 5}


# --------------------------------------------------------- spread in group

def test_spread_covers_every_episode_before_repeating_any():
    df = pd.DataFrame({"episode": [1, 1, 1, 1, 2, 2, 3, 3, 3]})
    got = hn.spread_within_group(df, 3)
    assert got["episode"].nunique() == 3        # one from each, not 3 from ep 1


def test_spread_takes_a_second_pass_only_after_a_first():
    df = pd.DataFrame({"episode": [1, 1, 1, 2, 2, 3]})
    got = hn.spread_within_group(df, 5)
    counts = got["episode"].value_counts()
    assert len(got) == 5
    assert counts.max() - counts.min() <= 1     # as even as the supply allows


def test_spread_is_deterministic_for_a_seed():
    df = pd.DataFrame({"episode": [1] * 10 + [2] * 10})
    a = hn.spread_within_group(df, 6, rng=np.random.default_rng(1))
    b = hn.spread_within_group(df, 6, rng=np.random.default_rng(1))
    assert a.index.tolist() == b.index.tolist()


def test_spread_asking_for_more_than_exists_returns_all():
    df = pd.DataFrame({"episode": [1, 2, 3]})
    assert len(hn.spread_within_group(df, 99)) == 3


# --------------------------------------------------------------- selection

def test_selection_balances_across_stations():
    m = _matched({"IPA4ST": (100, 2370), "A": (100, 85), "B": (100, 85),
                  "C": (100, 85)})
    sel = hn.select(m, total=300)
    counts = sel["site"].value_counts()
    # IPA4ST holds 90% of the false positives but takes nowhere near 90%.
    assert counts["IPA4ST"] / len(sel) < 0.4
    assert set(counts.index) == {"IPA4ST", "A", "B", "C"}


def test_held_out_stations_contribute_nothing():
    m = _matched({"IPA4ST": (100, 2370), "A": (100, 85), "B": (100, 85)})
    sel = hn.select(m, total=300, holdout=["A"])
    assert "A" not in set(sel["site"])
    assert len(sel) == 300


def test_only_false_positives_are_mined():
    m = _matched({"A": (50, 50)})
    sel = hn.select(m, total=100)
    assert (sel["verdict"] == "false_positive").all()
    assert len(sel) == 50           # only the 50 that exist


def test_selection_spreads_across_episodes_within_a_station():
    m = _matched({"A": (0, 100)}, episodes_per_station=10)
    sel = hn.select(m, total=10)
    assert sel["episode"].nunique() == 10       # one per episode, not 10 from one


def test_per_episode_cap_limits_one_bout():
    m = _matched({"A": (0, 100)}, episodes_per_station=5)
    sel = hn.select(m, total=100, per_episode_cap=2)
    assert len(sel) == 10                        # 5 episodes x 2
    assert sel.groupby("episode").size().max() == 2


def test_per_station_cap_is_respected():
    m = _matched({"A": (0, 500), "B": (0, 500)})
    sel = hn.select(m, total=1000, per_station_cap=50)
    assert sel["site"].value_counts().max() == 50


def test_selection_is_deterministic_for_a_seed():
    m = _matched({"A": (0, 200), "B": (0, 200)})
    a = hn.select(m, total=50, seed=7)
    b = hn.select(m, total=50, seed=7)
    assert a.index.tolist() == b.index.tolist()
    assert hn.select(m, total=50, seed=8).index.tolist() != a.index.tolist()


def test_mined_from_records_the_source_station():
    m = _matched({"A": (0, 50), "B": (0, 50)})
    sel = hn.select(m, total=20)
    assert (sel["mined_from"] == sel["site"]).all()


def test_empty_input_returns_empty():
    empty = pd.DataFrame(columns=["site", "verdict", "start_s"])
    assert len(hn.select(empty, total=100)) == 0


# -------------------------------------------------------------- the plan

def test_plan_shows_what_would_be_taken_and_what_is_held_back():
    m = _matched({"IPA4ST": (100, 2370), "A": (100, 85), "B": (100, 85)})
    p = hn.plan(m, total=300, holdout=["B"]).set_index("site")

    assert bool(p.loc["B", "held_out"]) is True
    assert p.loc["B", "selected"] == 0
    # The dominant station is deliberately under-used; the small one exhausted.
    assert p.loc["IPA4ST", "share_taken"] < 0.2
    assert p.loc["A", "share_taken"] == 1.0
    assert p.loc["TOTAL", "selected"] == 300


def test_summary_warns_when_nothing_is_held_out():
    m = _matched({"A": (10, 50), "B": (10, 50)})
    txt = hn.summarise_text(hn.plan(m, total=20))
    assert "NONE" in txt and "--holdout" in txt


def test_summary_names_the_held_out_stations():
    m = _matched({"A": (10, 50), "B": (10, 50)})
    txt = hn.summarise_text(hn.plan(m, total=20, holdout=["B"]), holdout=["B"])
    assert "B" in txt and "transfer test" in txt


# ---------------------------------------------------------- call recovery

def test_recovering_calls_takes_confirmed_calls_only():
    m = _matched({"A": (30, 70)})
    got = hn.recover_calls(m)
    assert len(got) == 30 and (got["verdict"] == "call").all()


def test_recovering_prefers_the_calls_the_model_was_least_sure_about():
    m = _matched({"A": (10, 10)})
    m.loc[m["verdict"] == "call", "confidence"] = [0.2, 0.3, 0.4, 0.5, 0.6,
                                                   0.95, 0.96, 0.97, 0.98, 0.99]
    got = hn.recover_calls(m, only_low_confidence=0.7)
    assert len(got) == 5
    assert got["confidence"].max() < 0.7


def test_recovering_can_be_capped_per_station():
    m = _matched({"A": (50, 10), "B": (50, 10)})
    got = hn.recover_calls(m, max_per_station=10)
    assert got.groupby("site").size().max() == 10


# ------------------------------------------------- the command-line wrapper

def _write_clips(rows, d):
    """Write a dummy WAV per row under the name extract_clips() would use."""
    import soundfile as sf
    os.makedirs(d, exist_ok=True)
    for _, r in rows.iterrows():
        name = (f"{r['recording']}__t{int(r['start_s']):05d}s"
                f"__conf{float(r['confidence']):.2f}.wav")
        sf.write(os.path.join(d, name), np.zeros(100), 44100)


def test_dry_run_prints_the_plan_and_copies_nothing(tmp_path):
    import subprocess
    m = _matched({"IPA4ST": (100, 2370), "A": (100, 85), "B": (100, 85)})
    p = tmp_path / "cleanup_vs_review.csv"
    m.to_csv(p, index=False)

    script = os.path.join(os.path.dirname(__file__), "..", "scripts",
                          "mine_hard_negatives.py")
    proc = subprocess.run(
        [sys.executable, script, "--matched", str(p), "--total", "300",
         "--holdout", "B", "--dry-run"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "transfer test" in proc.stdout
    assert "nothing copied" in proc.stdout
    assert not list(tmp_path.glob("**/hard_negatives"))


def test_it_copies_the_selected_clips_and_writes_a_manifest(tmp_path):
    import subprocess
    m = _matched({"A": (20, 60), "B": (20, 60)})
    p = tmp_path / "cleanup_vs_review.csv"
    m.to_csv(p, index=False)
    clips = tmp_path / "clips"
    _write_clips(m, str(clips))

    out = tmp_path / "mined"
    script = os.path.join(os.path.dirname(__file__), "..", "scripts",
                          "mine_hard_negatives.py")
    proc = subprocess.run(
        [sys.executable, script, "--matched", str(p), "--clips-dir", str(clips),
         "--out", str(out), "--total", "40"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    copied = list((out / "hard_negatives").glob("*.wav"))
    assert len(copied) == 40
    manifest = pd.read_csv(out / "mined_manifest.csv")
    assert len(manifest) == 40
    assert set(manifest["site"]) == {"A", "B"}
    assert (out / "mined_plan.csv").exists()


def test_recovered_calls_go_to_their_own_folder(tmp_path):
    import subprocess
    m = _matched({"A": (20, 40)})
    m.loc[m["verdict"] == "call", "confidence"] = [0.3] * 10 + [0.99] * 10
    p = tmp_path / "cleanup_vs_review.csv"
    m.to_csv(p, index=False)
    clips = tmp_path / "clips"
    _write_clips(m, str(clips))

    out = tmp_path / "mined"
    script = os.path.join(os.path.dirname(__file__), "..", "scripts",
                          "mine_hard_negatives.py")
    proc = subprocess.run(
        [sys.executable, script, "--matched", str(p), "--clips-dir", str(clips),
         "--out", str(out), "--total", "20", "--recover-calls",
         "--recover-below", "0.5"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    # Only the ten low-confidence calls, and they are kept apart from the
    # negatives so they cannot be folded into Background by mistake.
    assert len(list((out / "recovered_calls").glob("*.wav"))) == 10
    assert len(list((out / "hard_negatives").glob("*.wav"))) == 20
    rec = pd.read_csv(out / "recovered_manifest.csv")
    assert (rec["verdict"] == "call").all() and (rec["confidence"] < 0.5).all()


def test_missing_clip_files_are_reported_not_silently_dropped(tmp_path):
    import subprocess
    m = _matched({"A": (0, 40)})
    p = tmp_path / "cleanup_vs_review.csv"
    m.to_csv(p, index=False)
    clips = tmp_path / "clips"
    _write_clips(m.iloc[:10], str(clips))      # only a quarter exist

    out = tmp_path / "mined"
    script = os.path.join(os.path.dirname(__file__), "..", "scripts",
                          "mine_hard_negatives.py")
    proc = subprocess.run(
        [sys.executable, script, "--matched", str(p), "--clips-dir", str(clips),
         "--out", str(out), "--total", "40"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "WARNING" in proc.stdout
    assert "wrong folder" in proc.stdout
