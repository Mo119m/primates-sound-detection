"""
Aggregate the manual-review tables into the paper's field-deployment numbers.

Reviewers examine the exported detection clips in Kaleidoscope Pro (Wildlife
Acoustics) and produce one CSV per site. The reviewer's verdict lives in the
``MANUAL ID`` column: a false positive is tagged ``Noise`` and a confirmed call
is left blank (or tagged with the species). Each clip filename still encodes the
species, timestamp, start-second and model confidence, e.g.::

    Cernic__20210222T053000+0100_Short-term_Makokou__01540s__conf0.980.wav

This module reads any number of those CSVs, classifies each detection as a
confirmed call or a false positive, and tallies the per-site / per-species
numbers reported in the manuscript.

The one convention that must be set correctly is what a **blank** ``MANUAL ID``
means. By default a blank is treated as a *confirmed call* (the reviewer only
types a tag for the false positives). Set ``blank_is_confirmed=False`` if a
blank instead means "not yet reviewed".
"""
import os
import re
import glob
import pandas as pd

# MANUAL ID values (compared case-insensitively, trimmed) that mark a NON-call.
# Extend this set if a reviewer uses other tags for junk detections.
FALSE_POSITIVE_TAGS = {"noise", "noize", "n", "false", "fp", "junk", "unknown",
                       "insect", "bird", "wind", "rain"}

# <species>__<...timestamp...>__<start>s__conf<confidence>.wav
_FNAME_RE = re.compile(
    r"^(?P<species>.+?)__.*__(?P<start>\d+)s__conf(?P<conf>[0-9.]+)\.wav$"
)
# the recording portion, between the species prefix and the start-second field
_REC_RE = re.compile(r"^.+?__(?P<recording>.+)__\d+s__conf[0-9.]+\.wav$")
# Optional AudioMoth-style timestamp embedded in the recording part.
_TS_RE = re.compile(r"(\d{8}T\d{6})")


def _site_species_from_indir(indir, filename):
    """Infer (site, species) from the review row's INDIR path / filename.

    INDIR typically ends in ``.../<species>/<site>`` (e.g. ``Cernic/IPA1ST``).
    Falls back to the species encoded in the filename.
    """
    parts = [p for p in re.split(r"[\\/]", str(indir)) if p]
    site = parts[-1] if parts else ""
    species = parts[-2] if len(parts) >= 2 else ""
    m = _FNAME_RE.match(str(filename))
    if m and not species:
        species = m.group("species")
    return site, species


def load_review_csv(path, blank_is_confirmed=True):
    """
    Read one Kaleidoscope review CSV and return a tidy per-detection DataFrame.

    Columns: file, site, species, start_s, confidence, manual_id, verdict
    where verdict is 'call' (confirmed), 'false_positive', or 'unreviewed'.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    # Column headers vary between exports: case, spacing, and a trailing '*'
    # that Kaleidoscope adds to some columns ('IN FILE' vs 'IN FILE*'). Compare
    # on letters and digits only so every spelling maps to the same key.
    cols = {re.sub(r"[^A-Z0-9]", "", c.upper()): c for c in df.columns}
    infile_col = cols.get("INFILE") or cols.get("FILE")
    indir_col = cols.get("INDIR")
    manual_col = cols.get("MANUALID")
    if infile_col is None or manual_col is None:
        missing = [n for n, c in (("IN FILE", infile_col),
                                  ("MANUAL ID", manual_col)) if c is None]
        raise ValueError(f"{path}: could not find the {' and '.join(missing)} "
                         f"column(s). Found: {list(df.columns)}")

    rows = []
    for _, r in df.iterrows():
        fname = r[infile_col]
        indir = r[indir_col] if indir_col else ""
        site, species = _site_species_from_indir(indir, fname)
        m = _FNAME_RE.match(str(fname))
        species = species or (m.group("species") if m else "")
        start_s = int(m.group("start")) if m else -1
        conf = float(m.group("conf")) if m else float("nan")
        tsm = _TS_RE.search(str(fname))
        timestamp = tsm.group(1) if tsm else ""

        tag = str(r[manual_col]).strip()
        low = tag.lower()
        if low == "":
            verdict = "call" if blank_is_confirmed else "unreviewed"
        elif low in FALSE_POSITIVE_TAGS:
            verdict = "false_positive"
        else:
            # A non-empty, non-junk tag (e.g. a species code) = confirmed call.
            verdict = "call"

        rm = _REC_RE.match(str(fname))
        recording = rm.group("recording") if rm else ""

        rows.append({
            "file": fname, "site": site, "species": species,
            "recording": recording, "start_s": start_s, "confidence": conf,
            "timestamp": timestamp, "manual_id": tag, "verdict": verdict,
        })
    return pd.DataFrame(rows)


# Summary tables this pipeline writes. They often land next to the review CSVs,
# so skip them when scanning a folder rather than trying to parse them as
# reviews (they have no IN FILE / MANUAL ID columns).
GENERATED_CSVS = {
    "review_per_species.csv", "review_per_site.csv",
    "review_confirmed_by_site.csv", "review_all_detections.csv",
    "cleanup_vs_review.csv", "cleanup_eval_per_species.csv",
    "per_species_summary.csv", "per_station_summary.csv",
    "kept_calls_by_station.csv", "detection_labels.csv",
}


def load_review_dir(path_or_glob, blank_is_confirmed=True):
    """Load and concatenate every review CSV under a folder or glob pattern.

    Summary tables written by this pipeline are skipped, so re-running after a
    summary has been saved into the same folder still works.
    """
    # '~' is expanded by the shell but not by Python, so do it here.
    path_or_glob = os.path.expanduser(str(path_or_glob))
    if os.path.isdir(path_or_glob):
        files = sorted(glob.glob(os.path.join(path_or_glob, "**", "*.csv"),
                                 recursive=True))
    else:
        files = sorted(glob.glob(path_or_glob))
    files = [f for f in files if os.path.basename(f) not in GENERATED_CSVS]
    if not files:
        raise FileNotFoundError(f"No review CSVs matched {path_or_glob!r}")
    frames = [load_review_csv(f, blank_is_confirmed=blank_is_confirmed)
              for f in files]
    return pd.concat(frames, ignore_index=True)


def summarize(review_df):
    """
    Tally the field-deployment numbers from a per-detection review DataFrame.

    Returns a dict with:
      * ``per_species`` -- DataFrame: detections, confirmed, false_positive,
        unreviewed, precision
      * ``per_site``    -- DataFrame: detections per site x species
      * ``confirmed_by_site`` -- DataFrame: confirmed calls per site x species
      * ``totals``      -- dict: sites, species, detections, confirmed,
        false_positive, unreviewed
    """
    df = review_df

    def counts(sub):
        return pd.Series({
            "detections": len(sub),
            "confirmed": int((sub["verdict"] == "call").sum()),
            "false_positive": int((sub["verdict"] == "false_positive").sum()),
            "unreviewed": int((sub["verdict"] == "unreviewed").sum()),
        })

    per_species = df.groupby("species", sort=True).apply(counts) if len(df) else pd.DataFrame()
    if len(per_species):
        per_species["precision"] = [
            round(c / (c + f), 4) if (c + f) > 0 else pd.NA
            for c, f in zip(per_species["confirmed"], per_species["false_positive"])
        ]

    per_site = (df.pivot_table(index="site", columns="species", values="file",
                               aggfunc="count", fill_value=0)
                if len(df) else pd.DataFrame())
    conf_df = df[df["verdict"] == "call"]
    confirmed_by_site = (conf_df.pivot_table(index="site", columns="species",
                                             values="file", aggfunc="count",
                                             fill_value=0)
                         if len(conf_df) else pd.DataFrame())

    totals = {
        "sites": int(df["site"].nunique()) if len(df) else 0,
        "recordings": (int(df.loc[df["recording"] != "", "recording"].nunique())
                       if len(df) and "recording" in df.columns else 0),
        "species": int(df["species"].nunique()) if len(df) else 0,
        "detections": len(df),
        "confirmed": int((df["verdict"] == "call").sum()) if len(df) else 0,
        "false_positive": int((df["verdict"] == "false_positive").sum()) if len(df) else 0,
        "unreviewed": int((df["verdict"] == "unreviewed").sum()) if len(df) else 0,
    }
    return {"per_species": per_species, "per_site": per_site,
            "confirmed_by_site": confirmed_by_site, "totals": totals}


def report_text(review_df):
    """Plain-text summary ready to paraphrase into the manuscript."""
    s = summarize(review_df)
    t = s["totals"]
    out = ["FIELD-DEPLOYMENT SUMMARY (from manual-review CSVs)",
           "=" * 50,
           f"Sites            : {t['sites']}",
           f"Recordings       : {t['recordings']}",
           f"Total detections : {t['detections']}",
           f"Confirmed calls  : {t['confirmed']}",
           f"False positives  : {t['false_positive']}",
           f"Unreviewed       : {t['unreviewed']}", ""]
    ps = s["per_species"]
    if len(ps):
        out.append("Per species:")
        for sp, row in ps.iterrows():
            prec = row.get("precision")
            prec_s = f"{float(prec)*100:.1f}%" if pd.notna(prec) else "n/a"
            out.append(f"  {sp:18s} detections={int(row['detections']):5d}  "
                       f"confirmed={int(row['confirmed']):5d}  "
                       f"false={int(row['false_positive']):5d}  "
                       f"precision={prec_s}")
    return "\n".join(out)
