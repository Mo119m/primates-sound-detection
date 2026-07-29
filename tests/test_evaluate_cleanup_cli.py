"""Tests for scripts/evaluate_cleanup.py as a command line.

The one piece of logic the script owns (rather than delegating to
cleanup_eval) is --exclude-station: dropping a dominant station so a result can
be reported both with and without it. That has to actually restrict the
downstream numbers, and it must not overwrite the full run's CSVs.
"""

import os
import subprocess
import sys
import tempfile

import pandas as pd

REPO = os.path.join(os.path.dirname(__file__), "..")
SCRIPT = os.path.join(REPO, "scripts", "evaluate_cleanup.py")


def _two_station_case():
    """IPA1ST: 2 calls. IPA4ST: 4 false positives. Both flagged/clean mixed."""
    rev = tempfile.mkdtemp()
    rows = [
        ("IPA1ST", "Cernic__recA__00100s__conf0.9.wav", ""),
        ("IPA1ST", "Cernic__recA__00200s__conf0.8.wav", ""),
        ("IPA4ST", "Cernic__recB__00300s__conf0.7.wav", "Noise"),
        ("IPA4ST", "Cernic__recB__00400s__conf0.6.wav", "Noise"),
        ("IPA4ST", "Cernic__recB__00500s__conf0.5.wav", "Noise"),
        ("IPA4ST", "Cernic__recB__00600s__conf0.4.wav", "Noise"),
    ]
    with open(os.path.join(rev, "review.csv"), "w") as f:
        f.write('"INDIR","IN FILE","MANUAL ID"\n')
        for site, fname, mid in rows:
            f.write(f'"/x/Cernic/{site}","{fname}","{mid}"\n')

    cln = tempfile.mkdtemp()
    cols = ["species", "source_file", "start_time", "flag_reason"]
    pd.DataFrame([("Cernic", "recA.wav", 100.0, ""),
                  ("Cernic", "recB.wav", 300.0, "")],
                 columns=cols).to_csv(
        os.path.join(cln, "clean_detections.csv"), index=False)
    pd.DataFrame([("Cernic", "recA.wav", 200.0, "isolated"),
                  ("Cernic", "recB.wav", 400.0, "mahalanobis"),
                  ("Cernic", "recB.wav", 500.0, "isolated"),
                  ("Cernic", "recB.wav", 600.0, "mahalanobis")],
                 columns=cols).to_csv(
        os.path.join(cln, "suspicious_detections.csv"), index=False)
    return rev, cln


def _run(*args):
    proc = subprocess.run([sys.executable, SCRIPT, *args],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_full_run_covers_both_stations():
    rev, cln = _two_station_case()
    out = _run("--review", rev, "--cleanup", cln)
    assert "6" in out  # 6 detections total
    full = pd.read_csv(os.path.join(cln, "cleanup_vs_review.csv"))
    assert len(full) == 6


def test_exclude_station_restricts_the_numbers():
    rev, cln = _two_station_case()
    out = _run("--review", rev, "--cleanup", cln,
               "--exclude-station", "IPA4ST")
    assert "Excluding IPA4ST: dropped 4 detections, 2 remain" in out
    assert "1 stations" in out

    sub_dir = os.path.join(cln, "excluding_IPA4ST")
    sub = pd.read_csv(os.path.join(sub_dir, "cleanup_vs_review.csv"))
    assert len(sub) == 2
    assert set(sub["site"].unique()) == {"IPA1ST"}


def test_exclude_station_also_restricts_the_secondary_tables():
    """The confusion table and the per-species table are built from the
    evaluation dict, not from `matched`, so they have to be recomputed after the
    exclusion -- otherwise they silently keep reporting all stations."""
    rev, cln = _two_station_case()
    out = _run("--review", rev, "--cleanup", cln,
               "--exclude-station", "IPA4ST")

    # IPA1ST holds 2 calls and 0 false positives; the excluded station held all
    # 4 false positives. A stale table would still show them.
    conf = out.split("Manual verdict x cleanup verdict:")[1].split("Per species")[0]
    assert "4" not in conf, f"confusion table still reports excluded rows:\n{conf}"

    per_sp = pd.read_csv(os.path.join(cln, "excluding_IPA4ST",
                                      "cleanup_eval_per_species.csv"))
    assert int(per_sp["detections"].iloc[0]) == 2
    assert int(per_sp["false_positives"].iloc[0]) == 0


def test_exclude_station_does_not_clobber_the_full_run():
    rev, cln = _two_station_case()
    _run("--review", rev, "--cleanup", cln)
    _run("--review", rev, "--cleanup", cln, "--exclude-station", "IPA4ST")
    # The full run's CSV is still the full run.
    full = pd.read_csv(os.path.join(cln, "cleanup_vs_review.csv"))
    assert len(full) == 6


def test_unknown_station_name_warns_and_keeps_everything():
    rev, cln = _two_station_case()
    out = _run("--review", rev, "--cleanup", cln,
               "--exclude-station", "NOSUCHSTATION")
    assert "WARNING" in out
    assert "IPA4ST" in out  # the warning lists what is actually present
    sub = pd.read_csv(os.path.join(cln, "excluding_NOSUCHSTATION",
                                   "cleanup_vs_review.csv"))
    assert len(sub) == 6
