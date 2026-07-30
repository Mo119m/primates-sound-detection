"""
Estimate field recall from a small, exhaustively annotated sample of audio.

The manual review covered the detections, which measures precision: of the
sounds the detector reported, how many were genuine. It cannot measure recall,
because a call the detector never reported produced no clip to review. Recall
needs the opposite kind of listening -- take some continuous audio, write down
*every* call in it, and ask how many of those the detector found.

The expensive part is the listening, so the sampling has to be worth it:

* **Stratify by station.** Detector behaviour varies enormously between stations
  here (per-station precision ranges from 4 % to 93 %), so a sample drawn
  without regard to station would be dominated by whichever recordings happen to
  be longest.
* **Report an interval, not a number.** Recall from a handful of calls is a
  noisy estimate, and a bare percentage invites more confidence than a sample of
  that size supports. :func:`score_recall` returns a Wilson interval alongside
  the point estimate, and :func:`calls_needed_for_ci` says in advance how many
  calls a target precision requires.
* **Draw the sample before listening, and keep the plan.** Choosing segments
  after hearing them, or quietly discarding a segment that turned out to have no
  calls, biases the result upward. The plan is written to disk with its seed so
  the same sample can be regenerated and audited.

Nothing here needs the model: a plan is made from recording durations, and
scoring compares annotations against a detection table that already exists.
"""
import numpy as np
import pandas as pd

#: A call is counted as detected when a detection of the same species overlaps
#: it by at least this many seconds. Small but non-zero: a detection window that
#: merely abuts a call has not found it.
DEFAULT_MIN_OVERLAP_S = 0.5


def _site_col(df, site_col=None):
    if site_col:
        return site_col
    return "site" if "site" in df.columns else "station"


def plan_segments(recordings, n_segments=40, segment_s=300.0, seed=0,
                  site_col="site", min_duration_s=None):
    """
    Choose which stretches of audio to annotate exhaustively.

    ``recordings`` is a DataFrame with a station column, ``recording`` and
    ``duration_s``. Segments are allocated across stations as evenly as the
    available audio allows, then placed at random within randomly chosen
    recordings of that station.

    Returns one row per segment with the station, recording and the
    ``start_s``/``end_s`` to listen to, plus the ``seed`` used, so the plan is
    reproducible. Segments never overlap within a recording.
    """
    if not len(recordings):
        return pd.DataFrame()
    df = recordings.copy()
    sc = site_col if site_col in df.columns else _site_col(df)
    df["duration_s"] = pd.to_numeric(df["duration_s"], errors="coerce")
    floor = segment_s if min_duration_s is None else float(min_duration_s)
    df = df[df["duration_s"] >= floor]
    if not len(df):
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    stations = sorted(df[sc].astype(str).unique())
    # Even allocation, with the remainder spread over the first few stations
    # rather than all landing on one.
    base, extra = divmod(int(n_segments), len(stations))
    quota = {st: base + (1 if i < extra else 0) for i, st in enumerate(stations)}

    rows = []
    for st in stations:
        sub = df[df[sc].astype(str) == st]
        used = {}
        for _ in range(quota[st]):
            if not len(sub):
                break
            # Weight by duration: a recording twice as long has twice the audio
            # and should be twice as likely to be sampled.
            w = sub["duration_s"].to_numpy(dtype=float)
            idx = rng.choice(len(sub), p=w / w.sum())
            rec = sub.iloc[idx]
            name, dur = str(rec["recording"]), float(rec["duration_s"])

            placed = None
            for _attempt in range(20):
                start = float(rng.uniform(0.0, max(0.0, dur - segment_s)))
                end = start + segment_s
                if all(end <= s or start >= e for s, e in used.get(name, [])):
                    placed = (start, end)
                    break
            if placed is None:
                continue
            used.setdefault(name, []).append(placed)
            rows.append({sc: st, "recording": name,
                         "start_s": round(placed[0], 1),
                         "end_s": round(placed[1], 1),
                         "segment_s": segment_s, "seed": seed})

    out = pd.DataFrame(rows)
    return out.sort_values([sc, "recording", "start_s"]).reset_index(drop=True)


def annotation_template(plan_df, species=""):
    """
    An empty sheet for the annotator: one row per segment, to be expanded.

    The annotator adds one row per call heard, filling ``call_start_s`` and
    ``call_end_s`` in seconds from the start of the *recording* (not of the
    segment, which is easy to get wrong and impossible to check afterwards).
    A segment containing no calls keeps its single row with the times left blank
    -- that row is the evidence the segment was listened to, and dropping it
    would bias recall upward.
    """
    if not len(plan_df):
        return pd.DataFrame(columns=["site", "recording", "segment_start_s",
                                     "segment_end_s", "species",
                                     "call_start_s", "call_end_s", "note"])
    sc = _site_col(plan_df)
    return pd.DataFrame({
        sc: plan_df[sc].values,
        "recording": plan_df["recording"].values,
        "segment_start_s": plan_df["start_s"].values,
        "segment_end_s": plan_df["end_s"].values,
        "species": species,
        "call_start_s": "",
        "call_end_s": "",
        "note": "",
    })


def score_recall(annotations, detections, min_overlap_s=DEFAULT_MIN_OVERLAP_S,
                 window_s=2.0, site_col=None, species=None):
    """
    How many annotated calls the detector found.

    ``annotations`` holds the completed template: rows with a ``call_start_s``
    are calls, rows without are segments that turned out to be empty and are
    kept as evidence of effort. ``detections`` is any detection table with
    ``recording``, ``start_s`` (or ``start_time``) and ``species``.

    A call counts as detected when a detection of the same species in the same
    recording overlaps it by at least ``min_overlap_s``. Detections are treated
    as ``window_s`` long from their start.

    Returns a dict with the point estimate, a Wilson 95 % interval, the counts
    behind it, and a per-station breakdown. Only detections inside an annotated
    segment are considered, so detections elsewhere in the recording neither help
    nor hurt.
    """
    ann = annotations.copy()
    det = detections.copy()
    if "start_s" not in det.columns and "start_time" in det.columns:
        det["start_s"] = pd.to_numeric(det["start_time"], errors="coerce")
    if "recording" not in det.columns and "source_file" in det.columns:
        det["recording"] = (det["source_file"].astype(str)
                            .str.replace(r"\.wav$", "", regex=True))
    sc = _site_col(ann, site_col)

    for c in ("call_start_s", "call_end_s", "segment_start_s", "segment_end_s"):
        if c in ann.columns:
            ann[c] = pd.to_numeric(ann[c], errors="coerce")
    if species:
        ann = ann[ann.get("species", species).astype(str) == str(species)]
        det = det[det["species"].astype(str) == str(species)]

    calls = ann[ann["call_start_s"].notna()].copy()
    n_segments = len(ann[["recording", "segment_start_s"]].drop_duplicates())
    if not len(calls):
        return {"calls": 0, "detected": 0, "recall": None, "ci_low": None,
                "ci_high": None, "segments": n_segments,
                "audio_s": _audio_seconds(ann), "per_station": pd.DataFrame()}

    # A call with no end time is treated as one window long, which is the most
    # conservative reading: the smallest target a detection has to overlap.
    calls["call_end_s"] = calls["call_end_s"].fillna(
        calls["call_start_s"] + float(window_s))

    det_by_rec = {}
    for (rec, sp), sub in det.groupby(
            [det["recording"].astype(str), det["species"].astype(str)]):
        det_by_rec[(rec, sp)] = pd.to_numeric(
            sub["start_s"], errors="coerce").dropna().to_numpy(dtype=float)

    found = []
    for r in calls.itertuples():
        key = (str(r.recording), str(getattr(r, "species", "")))
        starts = det_by_rec.get(key)
        if starts is None or not len(starts):
            found.append(False)
            continue
        overlap = np.minimum(starts + float(window_s), r.call_end_s) - \
            np.maximum(starts, r.call_start_s)
        found.append(bool((overlap >= min_overlap_s).any()))
    calls["detected"] = found

    n, k = len(calls), int(calls["detected"].sum())
    lo, hi = wilson_interval(k, n)

    per = []
    if sc in calls.columns:
        for st, sub in calls.groupby(sc, sort=True):
            kk, nn = int(sub["detected"].sum()), len(sub)
            slo, shi = wilson_interval(kk, nn)
            per.append({sc: str(st), "calls": nn, "detected": kk,
                        "recall": round(kk / nn, 4),
                        "ci_low": slo, "ci_high": shi})

    return {"calls": n, "detected": k, "recall": round(k / n, 4),
            "ci_low": lo, "ci_high": hi, "segments": n_segments,
            "audio_s": _audio_seconds(ann),
            "per_station": pd.DataFrame(per), "annotated_calls": calls}


def _audio_seconds(ann):
    seg = ann[["recording", "segment_start_s", "segment_end_s"]].drop_duplicates()
    span = pd.to_numeric(seg["segment_end_s"], errors="coerce") - \
        pd.to_numeric(seg["segment_start_s"], errors="coerce")
    return float(span.dropna().sum())


def wilson_interval(k, n, z=1.96):
    """
    Wilson 95 % interval for a proportion.

    Preferred over the normal approximation because a recall sample is small and
    the estimate is often near 1, where the normal interval runs above 100 %
    and understates the uncertainty.
    """
    if not n:
        return None, None
    p = k / n
    d = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)


def calls_needed_for_ci(half_width=0.05, expected_recall=0.9, z=1.96):
    """
    How many annotated calls a given interval width needs.

    Answers the question worth asking *before* committing listening time: at an
    expected recall of ``expected_recall``, how many calls must the sample
    contain for the interval to be no wider than ±``half_width``. Uses the
    normal approximation, which is adequate for planning.
    """
    p = float(expected_recall)
    return int(np.ceil(z ** 2 * p * (1 - p) / float(half_width) ** 2))


def segments_needed(calls_needed, calls_per_hour, segment_s=300.0):
    """
    Translate a call target into segments and listening hours.

    ``calls_per_hour`` is the density the existing review implies for the
    stations being sampled -- the one input that has to come from the data rather
    than from a rule of thumb.
    """
    if calls_per_hour <= 0:
        return None
    hours = float(calls_needed) / float(calls_per_hour)
    return {"calls_needed": int(calls_needed),
            "audio_hours": round(hours, 2),
            "segments": int(np.ceil(hours * 3600.0 / float(segment_s)))}
