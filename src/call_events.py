"""
Consolidate overlapping detection windows into call events.

Detection slides a 2 s window forward 1 s at a time, so a vocalization longer
than the stride is seen by several windows and reported several times. Per-species
non-maximum suppression removes only windows overlapping by more than the IoU
threshold: two consecutive windows, ``[0,2]`` and ``[1,3]``, overlap by
1 s of 3 s spanned, an IoU of 0.33, which is below the 0.5 default and so both
survive by design -- a boundary-straddling call must not be suppressed just
because its neighbour saw it too.

The consequence is that a detection count is a count of *windows*, not of calls.
That distinction matters twice:

* **Ecologically.** A study wants the number of vocalizations. A four-second
  roar seen by three windows is one call, not three, so a window count overstates
  vocal activity by a factor that depends on call duration.
* **For review effort.** Consecutive windows on one call are one listening
  decision, at a much finer scale than the listening episodes of
  :mod:`episode_features` -- an event is one call, an episode is one bout or one
  run of noise.

Merging is by adjacency in time, not by classifier output, so it needs no labels
and no audio: two detections of the same species in the same recording belong to
the same event when their windows touch or overlap.
"""
import numpy as np
import pandas as pd

try:
    from . import config
except ImportError:  # standalone / Colab
    import config

#: Windows whose start times are no further apart than this are one event. The
#: default is the window length, which is exactly the condition "the two windows
#: overlap or touch" for a forward-sliding window of that size.
DEFAULT_MAX_GAP_S = float(getattr(config, "WINDOW_SIZE", 2.0))


def _site_col(df, site_col=None):
    if site_col:
        return site_col
    return "site" if "site" in df.columns else "station"


def assign_events(det_df, max_gap_s=DEFAULT_MAX_GAP_S, site_col=None):
    """
    Label each detection with the call event it belongs to.

    Returns a copy of ``det_df`` with an integer ``event`` column, in the input's
    original row order. Detections of one species in one recording join the same
    event while consecutive start times differ by at most ``max_gap_s``.
    """
    df = det_df.copy()
    if not len(df) or "start_s" not in df.columns:
        df["event"] = pd.Series(dtype="int64")
        return df

    sc = _site_col(df, site_col)
    keys = [c for c in (sc, "recording", "species") if c in df.columns]
    order = df.sort_values(keys + ["start_s"], kind="stable").index

    start = pd.to_numeric(df.loc[order, "start_s"], errors="coerce")
    if keys:
        block = df.loc[order, keys]
        new_group = block.ne(block.shift()).any(axis=1)
    else:
        new_group = pd.Series(False, index=order)
        if len(order):
            new_group.iloc[0] = True
    df["event"] = (new_group | (start.diff() > max_gap_s)).cumsum().reindex(
        df.index).astype("int64")
    return df


def events(det_df, max_gap_s=DEFAULT_MAX_GAP_S, site_col=None,
           window_s=None):
    """
    One row per call event: where it is, how long it spans, how many windows saw
    it, and -- when the table carries manual verdicts -- how they were judged.

    ``duration_s`` is the span from the first window's start to the last
    window's end, so a single-window event has the window's own length rather
    than zero.
    """
    df = assign_events(det_df, max_gap_s=max_gap_s, site_col=site_col)
    if "event" not in df.columns or not len(df) or "start_s" not in df.columns:
        return pd.DataFrame()

    win = float(window_s if window_s is not None
                else getattr(config, "WINDOW_SIZE", 2.0))
    sc = _site_col(df, site_col)
    start = pd.to_numeric(df["start_s"], errors="coerce")
    has_verdict = "verdict" in df.columns

    rows = []
    for eid, sub in df.groupby("event", sort=True):
        s = start.loc[sub.index]
        row = {
            "event": int(eid),
            sc: sub[sc].iloc[0] if sc in sub.columns else "",
            "recording": sub["recording"].iloc[0] if "recording" in sub.columns else "",
            "species": sub["species"].iloc[0] if "species" in sub.columns else "",
            "start_s": float(s.min()),
            "end_s": float(s.max()) + win,
            "duration_s": float(s.max() - s.min()) + win,
            "windows": len(sub),
        }
        if "confidence" in sub.columns:
            row["max_confidence"] = float(pd.to_numeric(
                sub["confidence"], errors="coerce").max())
        if has_verdict:
            n_calls = int((sub["verdict"] == "call").sum())
            row["call_windows"] = n_calls
            # An event is a call if any of the windows that saw it was confirmed:
            # the reviewer judged the sound, and the windows are the same sound.
            row["verdict"] = ("call" if n_calls else "false_positive")
            row["mixed"] = 0 < n_calls < len(sub)
        rows.append(row)
    return pd.DataFrame(rows)


def event_effort(det_df, max_gap_s=DEFAULT_MAX_GAP_S, site_col=None,
                 window_s=None):
    """
    What consolidating windows into events changes, overall and per station.

    ``windows`` and ``events`` are the two counts; ``vs_per_window`` is the ratio.
    When verdicts are present, ``call_windows`` and ``call_events`` give the same
    contrast for genuine calls specifically -- the difference between them is how
    much a window count overstates the number of vocalizations.
    """
    ev = events(det_df, max_gap_s=max_gap_s, site_col=site_col,
                window_s=window_s)
    if not len(ev):
        return pd.DataFrame()
    sc = _site_col(ev, site_col)

    def summarise(sub, label):
        row = {
            "windows": int(sub["windows"].sum()),
            "events": len(sub),
            "windows_per_event": round(float(sub["windows"].mean()), 2),
            "vs_per_window": round(len(sub) / float(sub["windows"].sum()), 4),
            "longest_event_s": round(float(sub["duration_s"].max()), 1),
        }
        if "verdict" in sub.columns:
            calls = sub[sub["verdict"] == "call"]
            row["call_windows"] = int(sub.get("call_windows", 0).sum())
            row["call_events"] = len(calls)
            row["windows_per_call_event"] = (
                round(float(calls["windows"].mean()), 2) if len(calls) else None)
            row["inflation"] = (round(row["call_windows"] / len(calls), 2)
                                if len(calls) else None)
            row["mixed_events"] = int(sub.get("mixed", pd.Series(dtype=bool)).sum())
        return pd.Series(row, name=label)

    rows = [summarise(ev, "all stations")]
    if sc in ev.columns and ev[sc].nunique() > 1:
        for st, sub in ev.groupby(sc, sort=True):
            rows.append(summarise(sub, str(st)))
    return pd.DataFrame(rows)


def duration_summary(det_df, max_gap_s=DEFAULT_MAX_GAP_S, site_col=None,
                     window_s=None):
    """
    How long the consolidated events are, split by verdict when available.

    This is what says whether merging is doing real work: if genuine calls span
    several windows and false positives do not, then a window count is biased
    towards calls, and vice versa.
    """
    ev = events(det_df, max_gap_s=max_gap_s, site_col=site_col,
                window_s=window_s)
    if not len(ev):
        return pd.DataFrame()

    groups = [("all events", ev)]
    if "verdict" in ev.columns:
        for v, sub in ev.groupby("verdict", sort=True):
            groups.append((str(v), sub))

    rows = {}
    for label, sub in groups:
        d = pd.to_numeric(sub["duration_s"], errors="coerce").dropna()
        w = pd.to_numeric(sub["windows"], errors="coerce").dropna()
        if not len(d):
            continue
        rows[label] = {
            "events": len(sub),
            "median_duration_s": round(float(d.median()), 1),
            "mean_duration_s": round(float(d.mean()), 1),
            "p90_duration_s": round(float(np.percentile(d, 90)), 1),
            "max_duration_s": round(float(d.max()), 1),
            "mean_windows": round(float(w.mean()), 2),
            "single_window_pct": round(100.0 * float((w == 1).mean()), 1),
        }
    return pd.DataFrame(rows).T
