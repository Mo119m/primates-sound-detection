"""
Choose which reviewed detections to fold back into training.

The pipeline already mines hard negatives, but it mines the ones its own filters
flagged. That is circular -- the model scores its own errors -- and it reaches
only the errors the filters happen to catch, 697 of 3 654 here. Once a manual
review exists, both problems go away: every confirmed false positive is a hard
negative, established independently of the model.

What replaces them is a selection problem, and it is the one that decides whether
retraining generalises. The false positives are wildly unevenly distributed --
one station supplied 2 370 of 3 654 here, because an untrained species was
calling there -- so mining them in proportion teaches the model that station's
intruder and very little else. That failure is invisible in a pooled score and
obvious the moment the gain is measured at stations the negatives did not come
from (see :mod:`field_gate`).

Three things this module does about it:

* **Balance across stations.** Quotas are filled by water-filling: an equal share
  to each station, with whatever a small station cannot supply redistributed
  among the rest, repeated until stable. Every station's error mode gets into
  training; none of them drowns the others.
* **Hold stations out on purpose.** Stations named as held out contribute
  nothing, so the retrained model can be tested on error modes it has provably
  never seen. Without this there is no honest transfer test, because every
  station's noise would be in the training set.
* **Spread within a station.** Detections are taken round-robin across listening
  episodes, so a quota is not spent on one continuous minute of the same sound.
  Two hundred clips of one bout teach far less than two hundred clips from two
  hundred different bouts.

It also selects **confirmed calls** to recover into the positive set. The review
found genuine calls the detector scored poorly, and folding those back is what
stops iterative mining from quietly narrowing the model onto the calls it already
finds easily -- the bias the manuscript's Limitations describes.
"""
import numpy as np
import pandas as pd


def _site_col(df, site_col=None):
    if site_col:
        return site_col
    return "site" if "site" in df.columns else "station"


def water_fill(available, total):
    """
    Split ``total`` across groups as evenly as their supply allows.

    ``available`` maps group -> how many items it can supply. Each group is
    offered an equal share; a group that cannot fill its share gives all it has
    and its shortfall is redistributed among those that can, repeated until no
    group is over-offered. The result is the most even allocation reachable,
    which is what keeps a station with 2 370 false positives from crowding out
    one with 42.

    Returns group -> count, summing to ``min(total, sum(available))``.
    """
    avail = {k: int(v) for k, v in available.items() if int(v) > 0}
    if not avail or total <= 0:
        return {k: 0 for k in available}

    total = int(min(total, sum(avail.values())))
    out = {k: 0 for k in avail}
    remaining, active = total, dict(avail)

    while remaining > 0 and active:
        share = remaining // len(active)
        if share == 0:
            # Fewer items left than groups: give one each, largest supply first,
            # so the tie-break is deterministic rather than dictionary order.
            for k in sorted(active, key=lambda k: (-active[k], str(k)))[:remaining]:
                out[k] += 1
                remaining -= 1
            break
        filled = []
        for k in sorted(active):
            take = min(share, active[k])
            out[k] += take
            active[k] -= take
            remaining -= take
            if active[k] == 0:
                filled.append(k)
        for k in filled:
            del active[k]

    return {k: out.get(k, 0) for k in available}


def spread_within_group(df, n, group_col="episode", rng=None):
    """
    Take ``n`` rows, round-robin across ``group_col``, to maximise diversity.

    One pass takes at most one row per episode, the next takes a second from the
    episodes that have one, and so on. A quota therefore covers as many distinct
    sounds as the data allows before it repeats any of them. Within an episode
    the pick is random rather than first-in-time, so a bout is not always
    represented by its opening seconds.
    """
    if n <= 0 or not len(df):
        return df.iloc[:0]
    if group_col not in df.columns:
        return df.sample(n=min(n, len(df)), random_state=0)

    rng = rng or np.random.default_rng(0)
    buckets = []
    for _, sub in df.groupby(group_col, sort=True):
        idx = sub.index.to_numpy().copy()   # groupby may hand back a read-only view
        rng.shuffle(idx)
        buckets.append(list(idx))

    picked, round_i = [], 0
    while len(picked) < n:
        added = False
        for b in buckets:
            if round_i < len(b):
                picked.append(b[round_i])
                added = True
                if len(picked) >= n:
                    break
        if not added:
            break
        round_i += 1
    return df.loc[picked]


def select(matched_df, verdict="false_positive", total=2000, holdout=(),
           site_col=None, per_station_cap=None, per_episode_cap=None,
           seed=0):
    """
    Choose the detections to fold into training.

    ``matched_df`` is the reviewed table (``verdict`` plus a station column).
    ``holdout`` names stations that must contribute nothing, so a retrained
    model can be tested on error modes it has never seen.

    Returns the selected rows with a ``mined_from`` column naming the station,
    ready to be copied by the caller. Selection is deterministic given ``seed``.
    """
    df = matched_df[matched_df["verdict"] == verdict].copy()
    sc = _site_col(df, site_col)
    if not len(df):
        return df

    held = {str(s) for s in holdout}
    if held and sc in df.columns:
        df = df[~df[sc].astype(str).isin(held)]
    if not len(df):
        return df

    if "episode" not in df.columns:
        try:
            from . import episode_features
        except ImportError:
            import episode_features
        if "start_s" in df.columns:
            df = episode_features.assign_episodes(df, site_col=sc)

    available = df.groupby(df[sc].astype(str).to_numpy()).size().to_dict()
    if per_station_cap:
        available = {k: min(v, int(per_station_cap)) for k, v in available.items()}
    quota = water_fill(available, total)

    rng = np.random.default_rng(seed)
    picked = []
    for st, n in quota.items():
        if n <= 0:
            continue
        sub = df[df[sc].astype(str) == st]
        if per_episode_cap and "episode" in sub.columns:
            sub = pd.concat([
                g.sample(n=min(len(g), int(per_episode_cap)), random_state=seed)
                for _, g in sub.groupby("episode", sort=True)])
        picked.append(spread_within_group(sub, n, rng=rng))

    if not picked:
        return df.iloc[:0]
    out = pd.concat(picked).copy()
    out["mined_from"] = out[sc].astype(str)
    return out


def plan(matched_df, total=2000, holdout=(), site_col=None,
         per_station_cap=None, per_episode_cap=None, seed=0):
    """
    What the selection would take, per station, before any files are copied.

    ``share_taken`` is the fraction of a station's confirmed false positives that
    would be used, and it is the column to read: a station near 1.0 has been
    exhausted, while one far below it is being deliberately held back so it
    cannot dominate.
    """
    sel = select(matched_df, total=total, holdout=holdout, site_col=site_col,
                 per_station_cap=per_station_cap,
                 per_episode_cap=per_episode_cap, seed=seed)
    sc = _site_col(matched_df, site_col)
    fps = matched_df[matched_df["verdict"] == "false_positive"]
    held = {str(s) for s in holdout}

    rows = []
    for st, sub in fps.groupby(fps[sc].astype(str).to_numpy(), sort=True):
        taken = int((sel[sc].astype(str) == st).sum()) if len(sel) else 0
        eps = (sel[sel[sc].astype(str) == st]["episode"].nunique()
               if len(sel) and "episode" in sel.columns else None)
        rows.append({
            sc: st,
            "false_positives": len(sub),
            "held_out": st in held,
            "selected": taken,
            "share_taken": round(taken / len(sub), 3) if len(sub) else 0.0,
            "episodes_covered": eps,
        })
    out = pd.DataFrame(rows)
    if len(out):
        out.loc[len(out)] = {sc: "TOTAL",
                             "false_positives": int(out["false_positives"].sum()),
                             "held_out": "", "selected": int(out["selected"].sum()),
                             "share_taken": "", "episodes_covered": ""}
    return out


def recover_calls(matched_df, max_per_station=None, only_low_confidence=None,
                  site_col=None, seed=0):
    """
    Confirmed calls to fold back into the positive set.

    Iterative mining adds the model's mistakes and nothing else, which narrows it
    over time onto the calls it already finds easily: a genuine call the model
    scored poorly is never contradicted, only its neighbours in feature space get
    reinforced. Recovering confirmed calls is what breaks that loop, and the ones
    worth recovering most are exactly the ones the model was least sure about --
    pass ``only_low_confidence`` to take those below a confidence.
    """
    df = matched_df[matched_df["verdict"] == "call"].copy()
    if not len(df):
        return df
    if only_low_confidence is not None and "confidence" in df.columns:
        conf = pd.to_numeric(df["confidence"], errors="coerce")
        df = df[conf < float(only_low_confidence)]
    if max_per_station:
        sc = _site_col(df, site_col)
        df = pd.concat([
            g.sample(n=min(len(g), int(max_per_station)), random_state=seed)
            for _, g in df.groupby(df[sc].astype(str).to_numpy(), sort=True)])
    return df


def summarise_text(plan_df, holdout=(), site_col=None):
    """A short report of a mining plan, for printing."""
    if not len(plan_df):
        return "Nothing to mine."
    sc = _site_col(plan_df, site_col)
    body = plan_df[plan_df[sc] != "TOTAL"]
    total = plan_df[plan_df[sc] == "TOTAL"].iloc[0]
    mined = body[~body["held_out"].astype(bool)]
    lines = [
        f"Confirmed false positives available : {int(total['false_positives'])}",
        f"Selected for training              : {int(total['selected'])} "
        f"across {int((body['selected'] > 0).sum())} stations",
    ]
    if len(mined) and mined["selected"].sum():
        shares = pd.to_numeric(mined["share_taken"], errors="coerce")
        lines.append(f"Share used, per mined station      : "
                     f"{shares.min():.1%} to {shares.max():.1%}")
    held = [str(r[sc]) for _, r in body.iterrows() if r["held_out"]]
    if held:
        n_held = int(body[body["held_out"].astype(bool)]["false_positives"].sum())
        lines.append(f"Held out (contribute nothing)      : "
                     f"{', '.join(held)}  -- {n_held} false positives kept back "
                     f"as the transfer test")
    else:
        lines.append("Held out                           : NONE. Every station's "
                     "noise will be in training, so there is no station left to "
                     "test transfer on -- pass --holdout.")
    return "\n".join(lines)
