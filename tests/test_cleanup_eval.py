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


def _cleanup_dir_full(clean_rows, susp_rows):
    """rows: (species, source_file, start_time, mahal, yamnet, isolated,
    yamnet_top, yamnet_score)."""
    import tempfile as _t
    d = _t.mkdtemp()
    cols = ["species", "source_file", "start_time", "flag_mahal", "flag_yamnet",
            "flag_isolated", "yamnet_top", "yamnet_score"]
    pd.DataFrame(clean_rows, columns=cols).to_csv(
        os.path.join(d, "clean_detections.csv"), index=False)
    pd.DataFrame(susp_rows, columns=cols).to_csv(
        os.path.join(d, "suspicious_detections.csv"), index=False)
    return d


def test_requiring_agreement_loses_fewer_calls():
    """Demanding two filters agree must discard no more than a single flag
    does, so it can never lose more genuine calls."""
    rows, clean, susp = [], [], []
    # a call flagged by one filter only -> survives a 2-vote rule
    rows.append(("Cernic__recA__00100s__conf0.9.wav", ""))
    susp.append(("Cernic", "recA.wav", 100.0, False, True, False, "Cricket", 0.2))
    # an FP flagged by two filters -> discarded under either rule
    rows.append(("Cernic__recA__00200s__conf0.9.wav", "Noise"))
    susp.append(("Cernic", "recA.wav", 200.0, True, True, False, "Insect", 0.8))
    # a clean call
    rows.append(("Cernic__recA__00300s__conf0.9.wav", ""))
    clean.append(("Cernic", "recA.wav", 300.0, False, False, False, "Animal", 0.4))

    matched, _ = cleanup_eval.run(_review_csv(rows), _cleanup_dir_full(clean, susp))
    va = cleanup_eval.vote_analysis(matched)

    one, two = va.loc[">= 1 filter(s) agree"], va.loc[">= 2 filter(s) agree"]
    assert one["calls_lost"] == 1 and two["calls_lost"] == 0
    assert two["fps_removed"] == 1          # the 2-flag FP still goes
    assert va.loc["no cleanup", "calls_lost"] == 0


def test_yamnet_score_threshold_can_suppress_low_confidence_flags():
    rows, clean, susp = [], [], []
    # genuine call flagged by a low-confidence YAMNet guess
    rows.append(("Cernic__recA__00100s__conf0.9.wav", ""))
    susp.append(("Cernic", "recA.wav", 100.0, False, True, False, "Cricket", 0.15))
    # false positive flagged by a confident YAMNet prediction
    rows.append(("Cernic__recA__00200s__conf0.9.wav", "Noise"))
    susp.append(("Cernic", "recA.wav", 200.0, False, True, False, "Insect", 0.85))

    matched, _ = cleanup_eval.run(_review_csv(rows), _cleanup_dir_full(clean, susp))
    ys = cleanup_eval.yamnet_score_sweep(matched, thresholds=(0.0, 0.5))

    # trusting every prediction costs the genuine call
    assert ys.loc["+ yamnet (score >= 0.0)", "calls_lost"] == 1
    # requiring confidence keeps it while still removing the false positive
    assert ys.loc["+ yamnet (score >= 0.5)", "calls_lost"] == 0
    assert ys.loc["+ yamnet (score >= 0.5)", "fps_removed"] == 1


def test_confidence_baseline_spends_the_same_budget():
    """The baseline must discard exactly as many detections as the cleanup, so
    the two precisions are comparable."""
    rows, clean, susp = [], [], []
    # flagged detections are false positives but carry HIGH confidence, so a
    # confidence rule cannot find them -- the filters should win here
    for s, conf in ((100, 0.99), (200, 0.98)):
        rows.append((f"Cernic__recA__{s:05d}s__conf{conf}.wav", "Noise"))
        susp.append(("Cernic", "recA.wav", float(s), True, False, False,
                     "Insect", 0.9))
    # genuine calls with low confidence: the confidence rule would drop these
    for s, conf in ((300, 0.41), (400, 0.42)):
        rows.append((f"Cernic__recA__{s:05d}s__conf{conf}.wav", ""))
        clean.append(("Cernic", "recA.wav", float(s), False, False, False,
                      "Animal", 0.3))

    matched, _ = cleanup_eval.run(_review_csv(rows), _cleanup_dir_full(clean, susp))
    cb = cleanup_eval.confidence_baseline(matched)

    assert len(cb) == 2
    # identical budget
    assert cb.iloc[0]["reviewed_after"] == cb.iloc[1]["reviewed_after"]
    # the filters removed both false positives; confidence removed both calls
    assert cb.iloc[0]["fps_removed"] == 2 and cb.iloc[0]["calls_lost"] == 0
    assert cb.iloc[1]["calls_lost"] == 2
    assert cb.iloc[0]["advantage"] > 0


def _graded_fixture():
    """Calls sit close to the training cluster with many neighbours; false
    positives sit far away and alone."""
    rows, clean, susp = [], [], []
    for i in range(10):                                   # genuine calls
        s = 100 + i
        rows.append((f"Cernic__recA__{s:05d}s__conf0.9.wav", ""))
        clean.append(("Cernic", "recA.wav", float(s), False, False, False,
                      "Animal", 0.3, 100.0 + i, 8))
    for i in range(10):                                   # false positives
        s = 500 + i
        rows.append((f"Cernic__recA__{s:05d}s__conf0.9.wav", "Noise"))
        clean.append(("Cernic", "recA.wav", float(s), False, False, False,
                      "Insect", 0.3, 9000.0 + i, 0))
    d = tempfile.mkdtemp()
    cols = ["species", "source_file", "start_time", "flag_mahal", "flag_yamnet",
            "flag_isolated", "yamnet_top", "yamnet_score", "mahalanobis_d2",
            "n_neighbours"]
    pd.DataFrame(clean, columns=cols).to_csv(
        os.path.join(d, "clean_detections.csv"), index=False)
    pd.DataFrame([], columns=cols).to_csv(
        os.path.join(d, "suspicious_detections.csv"), index=False)
    return _review_csv(rows), d


def test_signal_sweep_finds_the_separating_cutoff():
    rev, cln = _graded_fixture()
    matched, _ = cleanup_eval.run(rev, cln)
    sw = cleanup_eval.signal_sweep(matched, "mahalanobis_d2", flag_when="high")
    assert len(sw)
    best = sw.iloc[0]
    # a cutoff between the two groups removes every false positive, no calls
    assert best["calls_lost"] == 0
    assert best["fps_removed"] == 10
    assert best["precision"] == 1.0


def test_optimizer_respects_the_call_retention_floor():
    rev, cln = _graded_fixture()
    matched, _ = cleanup_eval.run(rev, cln)
    opt = cleanup_eval.optimize_thresholds(matched, min_call_retention=0.95)
    assert len(opt)
    # every returned configuration honours the floor
    assert (opt["calls_kept"] >= 0.95 * 10).all()
    assert opt.iloc[0]["precision"] >= 0.5     # and beats doing nothing


def test_holdout_sweep_scores_on_stations_it_did_not_tune_on():
    """The cutoff is chosen on half the stations and reported on the other
    half, so the evaluation row is never fitted to its own data."""
    rows, clean = [], []
    cols = ["species", "source_file", "start_time", "flag_mahal", "flag_yamnet",
            "flag_isolated", "yamnet_top", "yamnet_score", "mahalanobis_d2",
            "n_neighbours"]
    # four stations; in each, calls are confident and false positives are not
    for si, site in enumerate(["IPA1ST", "IPA2ST", "IPA3ST", "IPA4ST"]):
        for i in range(6):
            s = 100 * si + i
            rows.append((f"Cernic__rec{si}__{s:05d}s__conf0.99.wav", "", site))
            clean.append(("Cernic", f"rec{si}.wav", float(s), False, False,
                          False, "Animal", 0.3, 100.0, 5))
        for i in range(6):
            s = 100 * si + 50 + i
            rows.append((f"Cernic__rec{si}__{s:05d}s__conf0.50.wav", "Noise", site))
            clean.append(("Cernic", f"rec{si}.wav", float(s), False, False,
                          False, "Insect", 0.3, 100.0, 5))

    d = tempfile.mkdtemp()
    with open(os.path.join(d, "review.csv"), "w") as f:
        f.write('"INDIR","IN FILE","MANUAL ID"\n')
        for fname, mid, site in rows:
            f.write(f'"/x/Cernic/{site}","{fname}","{mid}"\n')
    cd = tempfile.mkdtemp()
    pd.DataFrame(clean, columns=cols).to_csv(
        os.path.join(cd, "clean_detections.csv"), index=False)
    pd.DataFrame([], columns=cols).to_csv(
        os.path.join(cd, "suspicious_detections.csv"), index=False)

    matched, _ = cleanup_eval.run(d, cd)
    hs = cleanup_eval.station_holdout_sweep(matched, "confidence",
                                            flag_when="low",
                                            min_call_retention=0.9)
    assert len(hs) == 3
    held = hs.iloc[1]          # the held-out row
    baseline = hs.iloc[2]      # same stations, no cleanup
    assert held["precision"] > baseline["precision"]
    assert held["calls_lost"] == 0        # confident calls all survive


def test_operating_points_respect_each_retention_floor():
    """Each row must honour its own floor, and demanding more retention can
    never allow higher precision than accepting less."""
    rows, clean = [], []
    cols = ["species", "source_file", "start_time", "flag_mahal", "flag_yamnet",
            "flag_isolated", "yamnet_top", "yamnet_score", "mahalanobis_d2",
            "n_neighbours"]
    # calls: close to the cluster, well connected in time
    for i in range(20):
        rows.append((f"Cernic__recA__{i:05d}s__conf0.9.wav", ""))
        clean.append(("Cernic", "recA.wav", float(i), False, False, False,
                      "Animal", 0.3, 100.0 + i, 10))
    # false positives: far from the cluster and isolated
    for i in range(20):
        s = 500 + i
        rows.append((f"Cernic__recA__{s:05d}s__conf0.9.wav", "Noise"))
        clean.append(("Cernic", "recA.wav", float(s), False, False, False,
                      "Insect", 0.3, 9000.0 + i, 0))

    d = tempfile.mkdtemp()
    pd.DataFrame(clean, columns=cols).to_csv(
        os.path.join(d, "clean_detections.csv"), index=False)
    pd.DataFrame([], columns=cols).to_csv(
        os.path.join(d, "suspicious_detections.csv"), index=False)

    matched, _ = cleanup_eval.run(_review_csv(rows), d)
    op = cleanup_eval.operating_points(matched, retention_levels=(0.95, 0.80))
    assert len(op) == 2
    for label, row in op.iterrows():
        floor = 0.95 if "95%" in label else 0.80
        assert row["calls_kept"] >= floor * 20
    # a separable signal reaches high precision even at the strict floor
    assert op.loc["keep >= 95% of calls", "precision"] > 0.9


def test_cross_validation_exposes_a_signal_that_only_works_in_sample():
    """A cutoff whose meaning differs per station looks good where it was tuned
    and fails on stations held out, which pooling must reveal."""
    rows, clean = [], []
    cols = ["species", "source_file", "start_time", "flag_mahal", "flag_yamnet",
            "flag_isolated", "yamnet_top", "yamnet_score", "mahalanobis_d2",
            "n_neighbours"]
    # Each station uses a different absolute scale for mahalanobis_d2, so no
    # single cutoff transfers; within a station calls sit below its noise.
    scales = {"A": 1.0, "B": 100.0, "C": 10000.0, "D": 1000000.0}
    for site, scale in scales.items():
        for i in range(10):
            rows.append((f"Cernic__rec{site}__{i:05d}s__conf0.9.wav", "", site))
            clean.append(("Cernic", f"rec{site}.wav", float(i), False, False,
                          False, "Animal", 0.3, scale, 5))
        for i in range(10):
            s = 100 + i
            rows.append((f"Cernic__rec{site}__{s:05d}s__conf0.9.wav", "Noise", site))
            clean.append(("Cernic", f"rec{site}.wav", float(s), False, False,
                          False, "Insect", 0.3, scale * 2, 5))

    d = tempfile.mkdtemp()
    with open(os.path.join(d, "review.csv"), "w") as f:
        f.write('"INDIR","IN FILE","MANUAL ID"\n')
        for fname, mid, site in rows:
            f.write(f'"/x/Cernic/{site}","{fname}","{mid}"\n')
    cd = tempfile.mkdtemp()
    pd.DataFrame(clean, columns=cols).to_csv(
        os.path.join(cd, "clean_detections.csv"), index=False)
    pd.DataFrame([], columns=cols).to_csv(
        os.path.join(cd, "suspicious_detections.csv"), index=False)

    matched, _ = cleanup_eval.run(d, cd)
    cv = cleanup_eval.station_cross_validation(matched, "mahalanobis_d2",
                                               flag_when="high")
    assert "POOLED (all held-out)" in cv.index
    pooled = cv.loc["POOLED (all held-out)", "precision"]
    baseline = cv.loc["POOLED, no cleanup", "precision"]
    # the per-station scales make the cutoff meaningless across stations
    assert pooled <= baseline
