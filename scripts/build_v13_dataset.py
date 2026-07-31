"""
Assemble the V13 training set, with a provable station for every clip.

V12's training set was built by folding auto-flagged detections back into
Background (the Step-5 hard-negative loop in the README). That loop had no
ground truth: the three cleanup filters decided what was a false positive, and
whatever they flagged became a negative. Cross-matching the flagged pool against
the 6 189 human-reviewed detections shows what that cost -- of the 368 flagged
clips that appear in the review, **68 are confirmed genuine calls**, 18.5 %.
Some carry confidence 0.99 and 1.00. The model was trained to reject the calls
it was most certain about, and 20 of those clips come from IPA19/IPA20, the two
stations the configuration says are held out.

So this script does not extend the old set; it rebuilds it from the labels that
were paid for. Three things change:

1. **Human verdicts replace filter guesses.** The 6 189 reviewed detections give
   2 535 confirmed calls and 3 654 confirmed false positives, each attached to a
   named station. Any auto-flagged clip contradicted by the review is corrected
   rather than dropped, so a genuine call moves to the Cernic class instead of
   silently disappearing.
2. **Every clip carries a station, or is marked unattributable.** Without this a
   leave-one-station-out evaluation is not possible and every reported gain is
   in-sample. Attribution runs off the lat/lon stamped into AudioMoth filenames;
   see ``station_of()`` for the five stations where that fails.
3. **The negative pool gains 16 000 bird segments** from the BirdNET run over the
   same site (external drive). Birds are what the high-frequency Cernic classes
   actually confuse, and the old Background class held 1 951 clips in total.

The output is a manifest CSV, not a folder of copies: the audio is ~3 GB and
already on disk in several places, and a manifest lets the training script
re-slice the same files per fold without moving anything.

Usage:
    python scripts/build_v13_dataset.py
    python scripts/build_v13_dataset.py --out data/outputs/v13_manifest.csv
    python scripts/build_v13_dataset.py --no-birdnet     # skip the external drive
"""
import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(REPO, "data")
DRIVE = "/Volumes/Gabon CNN"
BIRDNET = os.path.join(DRIVE, "Gabon BirdNET segments Birds")
RAW_AUDIO = os.path.join(DRIVE, "Gabon raw acoustic data National Park")

REVIEW_TABLE = os.path.join(DATA, "outputs/auto_cleanup/cleanup_vs_review.csv")
CLIPS_CERNIC = os.path.join(DATA, "outputs/detected_clips/Cernic")
CLIPS_COLOBUS = os.path.join(DATA, "outputs/detected_clips/Colobus_guereza")

# Filename shapes. AudioMoth writes the deployment lat/lon into the name, which
# is the only station evidence a clip carries once it leaves its folder.
COORD_RE = re.compile(r"([+-]\d+\.\d+[+-]\d+\.\d+)")
REVIEW_CLIP_RE = re.compile(r"^Cernic__(.+?)__(\d+)s__conf([\d.]+)\.wav$")
FLAGGED_CLIP_RE = re.compile(r"^Cernic__(.+?)__t(\d+)s__conf([\d.]+)\.wav$")
BIRDNET_RE = re.compile(r"^[\d.]+_\d+_(.+?)_([\d.]+)s_([\d.]+)s\.wav$")
STATION_IN_PATH_RE = re.compile(r"(ipa\d+)s?t?_", re.IGNORECASE)

# Fallback for when the external drive is not mounted and the coordinate map
# cannot be rebuilt: the stations that recorded without GPS in 2021-02.
NO_COORD_STATIONS = ["IPA1ST", "IPA2ST", "IPA4ST", "IPA6ST", "IPA7ST"]


def build_coord_map():
    """
    Map each deployment lat/lon to the station or stations that recorded it.

    Eleven of the sixteen stations stamp a unique coordinate into every filename.
    The other five (IPA1, IPA2, IPA4, IPA6, IPA7) were deployed with GPS off and
    write ``<timestamp>_Short-term_Makokou`` instead -- identical across all five,
    so a clip from one of them cannot be told from another by name alone. A
    further 50 files stamp ``+000.0000+000.0000`` (GPS never locked) and appear
    at both IPA13 and IPA20.

    The map keeps the ambiguous coordinates rather than discarding them, because
    a clip that is provably from one of *two* stations is still safely usable in
    the fourteen folds that hold out neither.
    """
    if not os.path.isdir(RAW_AUDIO):
        return {}, NO_COORD_STATIONS
    seen = {}
    no_coord = set()
    for station in sorted(os.listdir(RAW_AUDIO)):
        if not station.startswith("IPA"):
            continue
        for path in glob.glob(os.path.join(RAW_AUDIO, station, "*", "*.wav")):
            m = COORD_RE.search(os.path.basename(path))
            if m:
                seen.setdefault(m.group(1), set()).add(station)
            else:
                no_coord.add(station)
    return {c: sorted(s) for c, s in seen.items()}, sorted(no_coord)


def possible_stations(path, coord_map, no_coord_stations):
    """
    Every station a clip could have come from.

    One entry means the station is proven and the clip belongs to exactly one
    fold. Several entries mean the filename narrows it down but no further, and
    the clip must sit out each of those folds. An empty list means the clip is
    not from the 2021-02 sixteen-station deployment at all -- a curated reference
    clip, or audio from the 2022 Makokou sessions -- so no fold can leak through
    it and it is always safe to train on.

    Guessing a single station here would be the quietest possible way to inflate
    every held-out number in the paper, so nothing is guessed.
    """
    # A station named in the containing folder is direct evidence and wins.
    m = STATION_IN_PATH_RE.search(os.path.basename(os.path.dirname(path)))
    if m:
        return [m.group(1).upper() + "ST"]

    base = os.path.basename(path)
    m = COORD_RE.search(base)
    if m:
        return coord_map.get(m.group(1), [])

    # No coordinate. Only the 2021-02 deployment is at risk, and within it the
    # coordinate-less naming occurs at exactly the stations that had GPS off.
    if _period_of(path) == "2021-02" and "Short-term_Makokou" in base:
        return list(no_coord_stations)
    return []


def station_of(path, coord_map, no_coord_stations):
    """The station, when exactly one is possible; otherwise the empty string."""
    cands = possible_stations(path, coord_map, no_coord_stations)
    return cands[0] if len(cands) == 1 else ""


def _row(path, label, cands, source, verified, period=""):
    """
    One manifest row. ``cands`` is the list from ``possible_stations()``.

    ``station`` is filled only when a single station is proven, so downstream
    code that groups by station never sees a guess. ``possible_stations`` is
    what the fold builder must consult: a clip is excluded from a fold when the
    held-out station appears in it, whether or not ``station`` is set.
    """
    if isinstance(cands, str):
        cands = [cands] if cands else []
    return {
        "path": os.path.relpath(path, REPO),
        "label": label,
        "station": cands[0] if len(cands) == 1 else "",
        "possible_stations": ";".join(cands),
        "source": source,
        "verified": verified,
        "period": period or _period_of(path),
    }


def _period_of(path):
    m = re.search(r"(\d{4})(\d{2})\d{2}T", os.path.basename(path))
    return f"{m.group(1)}-{m.group(2)}" if m else ""


def collect_reference_clips(coord_map, no_coord):
    """
    The hand-curated reference clips, minus the negatives the review contradicts.

    These are the clips a person chose as examples of each class, so they are
    kept as-is with one exception: ``auto_flagged_fp`` was machine-labelled and
    is corrected against the review in ``apply_review()``.
    """
    rows = []
    spec = [
        ("species/CERNIC putty-nose 2s", "Cernic", True),
        ("species/CERNIC putty-nose 5s", "Cernic", True),
        ("species/CERNIC hacks", "Cernic", True),
        ("species/CERNIC keks", "Cernic", True),
        ("species/CERNIC pyows", "Cernic", True),
        ("species/CERNIC field_confirmed", "Cernic", True),
        ("species/Colobus guereza 2s windows", "Colobus_guereza", True),
        ("species/Colobus guereza Clips 5s", "Colobus_guereza", True),
        ("species/Colobus_confuser", "Colobus_confuser", True),
        ("background/background noise Clips 5sec", "Background", True),
        ("background/Cercocebus torquatus Clips 5s", "Background", True),
        ("background/Pan troglodytes Clips 5sec", "Background", True),
        ("background/wrong classified", "Background", True),
    ]
    for rel, label, verified in spec:
        for p in sorted(glob.glob(os.path.join(DATA, rel, "*.wav"))):
            rows.append(_row(p, label, possible_stations(p, coord_map, no_coord),
                             f"reference:{os.path.basename(rel)}", verified))

    # The auto-mined pool enters unverified and is corrected downstream.
    for p in sorted(glob.glob(os.path.join(
            DATA, "outputs/auto_cleanup/auto_flagged_fp", "**", "*.wav"),
            recursive=True)):
        rows.append(_row(p, "Background",
                         possible_stations(p, coord_map, no_coord),
                         "auto_flagged_fp", False))
    return pd.DataFrame(rows)


def collect_reviewed_detections():
    """
    The 6 189 reviewed C. nictitans detections, as labelled by the reviewer.

    ``verdict == 'call'`` becomes a Cernic positive and ``'false_positive'``
    becomes a negative. IPA4ST's negatives are tagged separately: 2 370 of the
    3 654 false positives are that one station, where an untrained species called
    in long bouts, and letting them into a generic Background class would let one
    station's intruder dominate the negative pool. The training script decides
    whether they form their own confuser output (the V12 Colobus_confuser
    pattern) or fold into Background; the manifest only records which they are.
    """
    if not os.path.exists(REVIEW_TABLE):
        print(f"  ! {REVIEW_TABLE} missing -- skipping reviewed detections")
        return pd.DataFrame()

    review = pd.read_csv(REVIEW_TABLE)
    index = {os.path.basename(p): p
             for p in glob.glob(os.path.join(CLIPS_CERNIC, "*", "*", "*.wav"))}

    rows, missing = [], 0
    for _, r in review.iterrows():
        path = index.get(r["file"])
        if path is None:
            missing += 1
            continue
        if r["verdict"] == "call":
            label, source = "Cernic", "review:confirmed_call"
        elif r["site"] == "IPA4ST":
            label, source = "Background", "review:ipa4st_intruder"
        else:
            label, source = "Background", "review:false_positive"
        rows.append(_row(path, label, [r["site"]], source, True))
    if missing:
        print(f"  ! {missing} reviewed rows had no exported clip")
    return pd.DataFrame(rows)


def collect_colobus_detections():
    """
    The 253 field Colobus detections, entered as negatives.

    The user reviewed these and found no genuine C. guereza among them -- they
    are thunder and other low-frequency noise. That is consistent with when they
    fire: 90.7 % fall between 19:00 and 05:00, and guereza roars at dawn.

    They are tagged ``colobus_field_fp`` rather than merged into the generic
    negatives so the decision stays reversible. The subset that most resembles a
    real roar (high low-frequency ratio, or timed with the dawn chorus) is
    written to a side file by ``--flag-suspect`` for the same reason: if a
    genuine roar is hiding in here, training it as a negative is the one mistake
    that cannot be undone by adding more data.
    """
    rows = []
    for p in sorted(glob.glob(os.path.join(CLIPS_COLOBUS, "*", "*", "*.wav"))):
        station = os.path.relpath(p, CLIPS_COLOBUS).split(os.sep)[0]
        rows.append(_row(p, "Background", [station], "colobus_field_fp", True))
    return pd.DataFrame(rows)


def collect_birdnet(coord_map, no_coord, review):
    """
    Bird segments from the BirdNET run, as domain-matched negatives.

    17 106 three-second segments over 251 species, cut from recordings made by
    the same hardware at the same site. Birds are the confuser the high-frequency
    Cernic call types actually lose to, and the whole Background class before
    this held 1 951 clips.

    The one real risk is a segment that contains a putty-nosed call BirdNET
    mislabelled as a bird: training that as Background repeats the exact mistake
    this rebuild is correcting. Segments whose recording and time span overlap a
    **confirmed call** are therefore dropped. Segments overlapping a confirmed
    false positive are kept -- those are hard negatives of the best kind. For
    recordings the review never covered no such check is possible; confirmed
    calls occupy roughly 0.3 % of recorded time, which bounds the residual risk.
    """
    if not os.path.isdir(BIRDNET):
        print(f"  ! {BIRDNET} not mounted -- skipping bird negatives")
        return pd.DataFrame()

    calls = review[review["verdict"] == "call"] if len(review) else pd.DataFrame()
    call_windows = {}
    for _, r in calls.iterrows():
        m = REVIEW_CLIP_RE.match(r["file"])
        if m:
            call_windows.setdefault(m.group(1), []).append(int(m.group(2)))

    rows, dropped = [], 0
    for p in sorted(glob.glob(os.path.join(BIRDNET, "*", "*.wav"))):
        m = BIRDNET_RE.match(os.path.basename(p))
        if not m:
            continue
        rec, start, end = m.group(1), float(m.group(2)), float(m.group(3))
        # A confirmed call is a 2 s window opening at its offset.
        if any(start < off + 2.0 and end > off
               for off in call_windows.get(rec, ())):
            dropped += 1
            continue
        species = os.path.basename(os.path.dirname(p))
        rows.append(_row(p, "Background",
                         possible_stations(p, coord_map, no_coord),
                         f"birdnet:{species}", False))
    print(f"  dropped {dropped} bird segments overlapping a confirmed call")
    return pd.DataFrame(rows)


def apply_review(manifest, review):
    """
    Correct the auto-flagged pool against the human verdicts.

    Matching is on **(recording, offset)** and deliberately ignores confidence.
    A clip and a reviewed detection that name the same recording and the same
    second are the same two seconds of audio; the confidence differs only
    because a different model version scored it, and the hard-negative loop ran
    across versions while the review was done once, on V12. Keying on confidence
    as well loses 19 clips that way -- 17 confirmed false positives and 2
    confirmed calls.

    The looser key is safe because it is not ambiguous here: across the 6 186
    (recording, second) pairs in the review, **none** carries two different
    verdicts, so no clip can be pulled toward a label the review does not
    support. That is checked rather than assumed, because recording names do
    repeat across the five stations that recorded without GPS.

    A flagged clip the review calls a genuine call is relabelled Cernic; one the
    review confirms is left a negative and marked verified, because a filter
    guess that a person checked is no longer a guess.
    """
    if not len(review):
        return manifest, 0

    truth, conflicts = {}, 0
    for _, r in review.iterrows():
        m = REVIEW_CLIP_RE.match(r["file"])
        if not m:
            continue
        key = (m.group(1), int(m.group(2)))
        if key in truth and truth[key][0] != r["verdict"]:
            conflicts += 1
            truth[key] = None          # ambiguous: refuse to label it
        elif key not in truth:
            truth[key] = (r["verdict"], r["site"])
    if conflicts:
        print(f"  ! {conflicts} (recording, second) pairs carry two verdicts "
              f"-- left unlabelled")

    recovered = 0
    for i, row in manifest[manifest["source"] == "auto_flagged_fp"].iterrows():
        m = FLAGGED_CLIP_RE.match(os.path.basename(row["path"]))
        if not m:
            continue
        hit = truth.get((m.group(1), int(m.group(2))))
        if hit is None:
            continue
        verdict, site = hit
        # The review names the station outright, which resolves clips whose
        # filename alone left several candidates.
        manifest.at[i, "station"] = site
        manifest.at[i, "possible_stations"] = site
        manifest.at[i, "verified"] = True
        if verdict == "call":
            manifest.at[i, "label"] = "Cernic"
            manifest.at[i, "source"] = "auto_flagged_fp:RECOVERED_CALL"
            recovered += 1
        else:
            manifest.at[i, "source"] = "auto_flagged_fp:confirmed_fp"
    return manifest, recovered


# Subfolders of auto_flagged_fp named after a filter rather than a station.
# Everything else in that tree was mined per station and hand-checked; these two
# are raw filter output that went into Background unread.
UNREVIEWED_FILTER_DUMPS = ("mahal", "yamnet")


def drop_unverified_filter_dumps(manifest):
    """
    Remove the filter dumps the review cannot vouch for.

    Cross-matching against the 6 189 reviewed detections puts a number on these
    two folders: of the clips that can be checked, 44 of 129 in ``mahal`` and
    24 of 239 in ``yamnet`` are **genuine calls** -- 34 % and 10 %. Neither
    filter is a reliable labeller and both were known not to be: YAMNet flags
    51.8 % of genuine calls (config.USE_YAMNET_FILTER is off for that reason),
    and Mahalanobis flags whatever sits far from the training distribution,
    which a loud unambiguous call often does.

    Clips the review reaches are kept with their human label -- confirmed false
    positives stay negatives, confirmed calls have already been moved to Cernic.
    The rest are dropped rather than guessed at: at a 19 % base rate of genuine
    calls, keeping them would put roughly 130 real calls into Background, which
    is the same mistake this rebuild exists to undo.
    """
    src = manifest["source"].fillna("")
    folder = manifest["path"].map(lambda p: os.path.basename(os.path.dirname(p)))
    suspect = (folder.isin(UNREVIEWED_FILTER_DUMPS)
               & src.str.startswith("auto_flagged_fp")
               & ~manifest["verified"].astype(bool))
    dropped = int(suspect.sum())
    return manifest[~suspect].reset_index(drop=True), dropped


def report(manifest):
    print("\n" + "=" * 72)
    print("V13 MANIFEST")
    print("=" * 72)
    print(f"\nTotal clips: {len(manifest)}\n")

    print("By label:")
    for label, n in manifest["label"].value_counts().items():
        v = manifest[(manifest.label == label) & manifest.verified]
        print(f"  {label:20s} {n:7d}   ({len(v)} human-verified)")

    print("\nBy source:")
    src = manifest["source"].str.split(":").str[0]
    for s, n in src.value_counts().items():
        print(f"  {s:28s} {n:7d}")

    cands = manifest["possible_stations"].fillna("")
    exact = (cands.str.count(";") == 0) & (cands != "")
    several = cands.str.contains(";")
    free = cands == ""
    print(f"\nStation provenance:")
    print(f"  exactly one station proven : {exact.sum():7d}  "
          f"(belongs to one fold)")
    print(f"  several stations possible  : {several.sum():7d}  "
          f"(sits out each of them)")
    print(f"  not from the 2021-02 sites : {free.sum():7d}  "
          f"(safe in every fold)")

    if several.sum():
        print("\n  Ambiguous groups:")
        for g, n in cands[several].value_counts().items():
            print(f"    {g}  {n}")

    field = manifest[exact]
    if len(field):
        print("\nBy station (proven clips only):")
        print(pd.crosstab(field["station"], field["label"]).to_string())

    print("\nSmallest training pool over the 16 folds:")
    stations = sorted(manifest.loc[exact, "station"].unique())
    sizes = []
    for s in stations:
        keep = ~cands.str.split(";").apply(lambda c, s=s: s in c)
        sizes.append((s, int(keep.sum()), int((~keep).sum())))
    for s, keep, drop in sorted(sizes, key=lambda x: x[1])[:3]:
        print(f"  hold out {s:8s} -> train on {keep} clips, {drop} withheld")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(DATA, "outputs/v13_manifest.csv"))
    ap.add_argument("--no-birdnet", action="store_true",
                    help="skip the external-drive bird segments")
    args = ap.parse_args()

    print("Mapping deployment coordinates to stations...")
    coord_map, no_coord = build_coord_map()
    unique = sum(1 for v in coord_map.values() if len(v) == 1)
    print(f"  {unique}/{len(coord_map)} coordinates resolve to exactly one station")
    print(f"  recorded without GPS (name is ambiguous): {', '.join(no_coord)}")

    review = (pd.read_csv(REVIEW_TABLE) if os.path.exists(REVIEW_TABLE)
              else pd.DataFrame())

    print("\nCollecting reference clips...")
    manifest = collect_reference_clips(coord_map, no_coord)
    print(f"  {len(manifest)}")

    print("Correcting the auto-flagged pool against the review...")
    manifest, recovered = apply_review(manifest, review)
    print(f"  recovered {recovered} genuine calls that were trained as Background")
    manifest, dropped = drop_unverified_filter_dumps(manifest)
    print(f"  dropped {dropped} unverifiable clips from the mahal/yamnet dumps")

    print("Collecting reviewed detections...")
    reviewed = collect_reviewed_detections()
    print(f"  {len(reviewed)}")

    print("Collecting field Colobus detections...")
    colobus = collect_colobus_detections()
    print(f"  {len(colobus)}")

    birds = pd.DataFrame()
    if not args.no_birdnet:
        print("Collecting BirdNET bird segments...")
        birds = collect_birdnet(coord_map, no_coord, review)
        print(f"  {len(birds)}")

    manifest = pd.concat([manifest, reviewed, colobus, birds], ignore_index=True)
    # A clip exported once and reviewed once can arrive from two collectors; the
    # reviewed copy carries the human label and wins.
    before = len(manifest)
    manifest = manifest.sort_values("verified", ascending=False)
    manifest = manifest.drop_duplicates(subset="path", keep="first")
    if before != len(manifest):
        print(f"\n  de-duplicated {before - len(manifest)} clips listed twice "
              f"(kept the human-labelled copy)")

    manifest = manifest.reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    manifest.to_csv(args.out, index=False)
    report(manifest)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
