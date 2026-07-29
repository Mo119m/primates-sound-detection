"""Tests for scripts/rank_signals_experiment.py.

The script exists to decide whether a candidate signal goes in the paper, so the
property that matters most is that it says *no* when the signal is noise. Both
directions are tested: a planted structure has to be found, and a random column
has to be rejected by the held-out selection even though it can look good
in-sample.
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd

REPO = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

os.environ.setdefault("PRIMATE_DATA_ROOT", "/tmp/test_data")

import episode_features  # noqa: E402
import rank_signals_experiment as rse  # noqa: E402

BASE = {"confidence": True, "softmax_margin": True,
        "n_neighbours": True, "mahalanobis_d2": False}


def _table(n_stations=4, dense_fp=True, seed=0):
    """Genuine calls in sparse bouts; false positives either in one dense run or
    in bouts of exactly the same shape.

    With ``dense_fp`` the temporal structure separates the two classes and the
    candidates should find it. Without it the false positives are laid out
    *identically* to the calls -- same bout size, same spacing, only later in the
    recording -- so every temporal feature is the same for both classes and there
    is genuinely nothing to find. Anything less than that is not a null: false
    positives spaced far enough apart to become singleton episodes would still be
    separable by episode size, which is structure, not noise.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for si in range(n_stations):
        st = f"IPA{si + 1}ST"
        for b in range(6):                                  # sparse bouts
            for k in range(4):
                rows.append(dict(site=st, species="Cernic", recording=f"rec{si}",
                                 start_s=b * 2000 + k * 40, verdict="call"))
        if dense_fp:
            for k in range(60):                             # one dense run
                rows.append(dict(site=st, species="Cernic", recording=f"rec{si}",
                                 start_s=30000 + k * 2,
                                 verdict="false_positive"))
        else:
            for b in range(15):                             # bouts of 4, as above
                for k in range(4):
                    rows.append(dict(site=st, species="Cernic",
                                     recording=f"rec{si}",
                                     start_s=30000 + b * 2000 + k * 40,
                                     verdict="false_positive"))
    df = pd.DataFrame(rows)
    # Per-clip signals carry no information here, so any gain must come from
    # the temporal candidates rather than from the baseline.
    df["cleanup"] = "clean"
    df["confidence"] = 0.9 + 0.05 * rng.random(len(df))
    df["softmax_margin"] = 0.85 + 0.1 * rng.random(len(df))
    df["n_neighbours"] = 4
    df["mahalanobis_d2"] = 200 + 50 * rng.random(len(df))
    return episode_features.add_episode_features(df)


def test_finds_a_planted_temporal_structure():
    df = _table(dense_fp=True)
    cands = episode_features.available_candidates(df)
    alone = rse.candidates_alone(df, cands, "site")

    # Dense runs are the false positives, so "low rate = genuine" must lead.
    top = alone.index[0]
    assert "episode_rate (low=genuine)" in alone.index
    assert alone.loc["episode_rate (low=genuine)", "mean_avg_precision"] > 0.9
    assert alone.loc["episode_rate (high=genuine)", "mean_avg_precision"] < 0.5
    assert alone.loc[top, "stations_above_arbitrary"] == 4


def test_held_out_selection_confirms_a_real_signal():
    df = _table(dense_fp=True)
    cands = episode_features.available_candidates(df)
    hs = rse.holdout_selection(df, cands, "site", BASE, max_added=2)

    assert hs["site"].iloc[-1] == "MEAN (held out)"
    assert float(hs["delta"].iloc[-1]) > 0.1
    # Every station improves, since the structure was planted at all of them.
    per_station = pd.to_numeric(hs["delta"].iloc[:-1], errors="coerce")
    assert (per_station > 0).all()


def test_held_out_selection_rejects_a_random_signal():
    """The negative control. A column of noise can win in-sample -- forward
    selection will happily pick it -- but it must not survive the held-out
    check, or the script would launder noise into the paper."""
    df = _table(dense_fp=False, seed=7)
    rng = np.random.default_rng(11)
    df["episode_rate"] = rng.random(len(df))          # replace with pure noise
    df["episode_mean_gap_s"] = rng.random(len(df))
    cands = {"episode_rate": False, "episode_mean_gap_s": True}

    hs = rse.holdout_selection(df, cands, "site", BASE, max_added=2)
    delta = float(hs["delta"].iloc[-1])
    assert delta < 0.02, f"noise passed the held-out check with delta={delta}"


def test_no_temporal_structure_means_no_gain():
    """Same detections, but false positives spread like the calls. The temporal
    candidates then have nothing to read and must not manufacture a gain."""
    df = _table(dense_fp=False, seed=3)
    cands = episode_features.available_candidates(df)
    hs = rse.holdout_selection(df, cands, "site", BASE, max_added=1)
    assert float(hs["delta"].iloc[-1]) < 0.05


def test_added_to_baseline_reports_the_baseline_itself():
    df = _table(dense_fp=True)
    cands = episode_features.available_candidates(df)
    added = rse.added_to_baseline(df, cands, "site", BASE)
    assert "(the four reported signals)" in added.index
    assert float(added.loc["(the four reported signals)", "delta"]) == 0.0
    # Deltas are measured against that row.
    best = added["mean_avg_precision"].astype(float).max()
    base = float(added.loc["(the four reported signals)", "mean_avg_precision"])
    assert abs(float(added["delta"].astype(float).max()) - (best - base)) < 1e-6


def test_runs_as_a_command_and_writes_its_csvs(tmp_path):
    df = _table(dense_fp=True)
    p = tmp_path / "cleanup_vs_review.csv"
    df.drop(columns=[c for c in episode_features.CANDIDATE_SIGNALS
                     if c in df.columns] + ["episode"]).to_csv(p, index=False)

    proc = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts",
                                      "rank_signals_experiment.py"),
         "--matched", str(p)],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "VERDICT:" in proc.stdout
    for name in ("rank_signals_alone.csv", "rank_signals_added.csv",
                 "rank_signals_holdout.csv"):
        assert (tmp_path / name).exists(), name


def test_exclude_station_is_honoured(tmp_path):
    df = _table(dense_fp=True)
    p = tmp_path / "cleanup_vs_review.csv"
    df.to_csv(p, index=False)
    proc = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts",
                                      "rank_signals_experiment.py"),
         "--matched", str(p), "--exclude-station", "IPA1ST"],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "across 3 stations" in proc.stdout
    hs = pd.read_csv(tmp_path / "rank_signals_holdout.csv")
    assert "IPA1ST" not in hs["site"].astype(str).tolist()
