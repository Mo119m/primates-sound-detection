"""
Manual annotation helpers for detection review.

After Step 5 exports one WAV per detection to
``data/outputs/detected_clips/<species>/<station>/``, a human listens to each
clip and marks it as a genuine call or a false positive. These helpers:

  * scan the exported clips and parse the species / recording / start-time /
    confidence encoded in each filename,
  * load and save the labels to a single resumable CSV (so you can stop and
    resume, and re-label a clip without creating duplicates), and
  * summarise the labels into the exact numbers reported in the paper's
    field-deployment section (per-station detection counts, Cernic detections
    retained after manual review, Colobus false-positive counts, precision).

The clip filename layout written by ``utils.extract_all_detected_clips`` is::

    <species>__<recording>__<00042>s__conf<0.873>.wav

and clips are grouped as ``<clips_dir>/<species>/<station>/<clip>.wav`` (the
station layer mirrors the source recording's folder; it is absent for a flat
single-folder run).

The UI (audio player, spectrogram, buttons) lives in
``main_pipeline_notebooks/annotate_detections.ipynb``; everything here is plain
data manipulation so it stays import-safe and testable without a notebook.
"""
import os
import re
import glob
import pandas as pd

try:
    from . import config
except ImportError:  # running as a standalone script (e.g. in a notebook)
    import config

# Labels a reviewer can assign. Values are deliberately NOT "true"/"false" so
# that pandas / Excel never coerce the CSV column to booleans.
TRUE_CALL = "call"            # a genuine target call (true positive)
FALSE_POS = "false_positive"  # a false positive (non-target sound)
UNSURE = "unsure"             # cannot tell -- revisit later
LABELS = (TRUE_CALL, FALSE_POS, UNSURE)

ANNOTATION_COLUMNS = ["clip_id", "label", "call_type", "note"]

# <species>__<recording>__<start>s__conf<confidence>.wav
_CLIP_RE = re.compile(
    r"^(?P<species>.+?)__(?P<recording>.+)__(?P<start>\d+)s__conf(?P<conf>[0-9.]+)\.wav$"
)


def default_clips_dir():
    """Where Step 5 writes the review clips."""
    return os.path.join(config.OUTPUT_ROOT, "detected_clips")


def default_annotation_csv():
    """Where this module stores the human labels."""
    return os.path.join(config.OUTPUT_ROOT, "annotations", "detection_labels.csv")


def scan_clips(clips_dir=None):
    """
    Scan every exported detection clip and return a DataFrame with one row per
    clip and the metadata parsed from its path and filename.

    Columns: clip_id (path relative to clips_dir -- the stable key used in the
    annotation CSV), path (absolute), species, station, recording, start_s,
    confidence.
    """
    if clips_dir is None:
        clips_dir = default_clips_dir()
    clips_dir = os.path.abspath(clips_dir)

    rows = []
    for path in sorted(glob.glob(os.path.join(clips_dir, "**", "*.wav"),
                                 recursive=True)):
        rel = os.path.relpath(path, clips_dir)
        parts = rel.split(os.sep)
        species_folder = parts[0] if len(parts) > 1 else "(unknown)"
        # Anything between the species folder and the file is the station path.
        station = os.sep.join(parts[1:-1]) if len(parts) > 2 else "(root)"

        m = _CLIP_RE.match(os.path.basename(path))
        if m:
            rows.append({
                "clip_id": rel,
                "path": path,
                "species": m.group("species"),
                "station": station,
                "recording": m.group("recording"),
                "start_s": int(m.group("start")),
                "confidence": float(m.group("conf")),
            })
        else:
            # Unparseable name: keep it so it can still be reviewed.
            rows.append({
                "clip_id": rel,
                "path": path,
                "species": species_folder,
                "station": station,
                "recording": "",
                "start_s": -1,
                "confidence": float("nan"),
            })

    df = pd.DataFrame(rows, columns=["clip_id", "path", "species", "station",
                                     "recording", "start_s", "confidence"])
    # Deterministic review order: species, then station, then time.
    if len(df):
        df = df.sort_values(["species", "station", "recording", "start_s"],
                            kind="stable").reset_index(drop=True)
    return df


def load_annotations(csv_path=None):
    """Load the labels CSV (empty frame with the right columns if none yet)."""
    if csv_path is None:
        csv_path = default_annotation_csv()
    if os.path.exists(csv_path):
        # Force every column to string so labels like "call"/"false_positive"
        # are never re-typed (pandas coerces bare true/false to booleans).
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        for col in ANNOTATION_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[ANNOTATION_COLUMNS]
    return pd.DataFrame(columns=ANNOTATION_COLUMNS)


def save_annotation(clip_id, label, call_type="", note="", csv_path=None):
    """
    Record one label and persist immediately (upsert -- re-labelling a clip
    overwrites its previous row rather than duplicating it). Returns the full
    updated annotations DataFrame.
    """
    if csv_path is None:
        csv_path = default_annotation_csv()
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}, got {label!r}")

    ann = load_annotations(csv_path)
    ann = ann[ann["clip_id"] != clip_id]  # drop any prior label for this clip
    new_row = pd.DataFrame([{
        "clip_id": clip_id, "label": label,
        "call_type": call_type or "", "note": note or "",
    }])
    ann = pd.concat([ann, new_row], ignore_index=True)

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    ann.to_csv(csv_path, index=False)
    return ann


def merge(clips_df, ann_df):
    """Join clips with their labels (unlabeled clips get label = NaN)."""
    return clips_df.merge(ann_df, on="clip_id", how="left")


def progress(clips_df, ann_df):
    """Return (labeled, total) so a notebook can show a progress bar."""
    total = len(clips_df)
    labeled = clips_df["clip_id"].isin(set(ann_df["clip_id"])).sum()
    return int(labeled), int(total)


def summarize(clips_df, ann_df):
    """
    Build the field-deployment tallies for the paper.

    Returns a dict with:
      * ``per_species``  -- DataFrame: detections, true, false, unsure, precision
      * ``per_station``  -- DataFrame: detections per station x species
      * ``true_by_station`` -- DataFrame: confirmed true calls per station x species
      * ``totals``       -- dict of headline numbers (stations, recordings,
                            detections, labeled, unlabeled)
    Precision = true / (true + false), ignoring unsure/unlabeled.
    """
    df = merge(clips_df, ann_df)

    def counts(sub):
        return pd.Series({
            "detections": len(sub),
            "true": int((sub["label"] == TRUE_CALL).sum()),
            "false": int((sub["label"] == FALSE_POS).sum()),
            "unsure": int((sub["label"] == UNSURE).sum()),
            "unlabeled": int(sub["label"].isna().sum()),
        })

    per_species = (df.groupby("species", sort=True).apply(counts)
                   if len(df) else pd.DataFrame())
    if len(per_species):
        per_species["precision"] = [
            round(t / (t + f), 4) if (t + f) > 0 else pd.NA
            for t, f in zip(per_species["true"], per_species["false"])
        ]

    per_station = (df.pivot_table(index="station", columns="species",
                                  values="clip_id", aggfunc="count", fill_value=0)
                   if len(df) else pd.DataFrame())

    true_df = df[df["label"] == TRUE_CALL]
    true_by_station = (true_df.pivot_table(index="station", columns="species",
                                           values="clip_id", aggfunc="count",
                                           fill_value=0)
                       if len(true_df) else pd.DataFrame())

    totals = {
        "stations": int(df["station"].nunique()) if len(df) else 0,
        "recordings": int(df["recording"].replace("", pd.NA).nunique()) if len(df) else 0,
        "detections": len(df),
        "labeled": int(df["label"].notna().sum()) if len(df) else 0,
        "unlabeled": int(df["label"].isna().sum()) if len(df) else 0,
    }

    return {
        "per_species": per_species,
        "per_station": per_station,
        "true_by_station": true_by_station,
        "totals": totals,
    }


def report_text(clips_df, ann_df):
    """
    Render a plain-text summary of the numbers the paper's field-deployment
    paragraph needs, ready to paraphrase into the manuscript.
    """
    s = summarize(clips_df, ann_df)
    t = s["totals"]
    lines = []
    lines.append("FIELD-DEPLOYMENT SUMMARY (for the paper)")
    lines.append("=" * 44)
    lines.append(f"Stations reviewed     : {t['stations']}")
    lines.append(f"Recordings with hits  : {t['recordings']}")
    lines.append(f"Total detections      : {t['detections']}")
    lines.append(f"Labeled / unlabeled   : {t['labeled']} / {t['unlabeled']}")
    if t["unlabeled"]:
        lines.append(f"  -> {t['unlabeled']} clips still need a label before the "
                     f"numbers are final.")
    lines.append("")
    lines.append("Per species (true = kept after manual review):")
    ps = s["per_species"]
    if len(ps):
        for sp, row in ps.iterrows():
            prec = row.get("precision")
            prec_s = f"{float(prec)*100:.1f}%" if pd.notna(prec) else "n/a"
            lines.append(f"  {sp:20s}  detections={int(row['detections']):5d}  "
                         f"true={int(row['true']):5d}  false={int(row['false']):5d}  "
                         f"unsure={int(row['unsure']):3d}  precision={prec_s}")
    else:
        lines.append("  (nothing labeled yet)")
    return "\n".join(lines)
