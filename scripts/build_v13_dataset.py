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
# External drive (LaCie Rugged Mini USB3, volume "Gabon CNN").
# Windows: "D:/"   WSL2: "/mnt/d"   macOS: "/Volumes/Gabon CNN"
DRIVE = "D:/"
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
        # The map is what proves which station a clip came from. Falling back to
        # the hardcoded list still marks the coordinate-less stations correctly,
        # but the eleven stations that DO stamp coordinates would go
        # unattributed, and unattributed clips are excluded from no fold at all
        # -- so every leave-one-station-out number would quietly include the
        # station it claims to hold out.
        print(f"  ! {RAW_AUDIO} not mounted: falling back to the hardcoded "
              f"no-GPS station list.\n"
              f"    Clips from the eleven coordinate-stamped stations will not "
              f"be attributable,\n    which makes leave-one-station-out unsafe. "
              f"Mount the drive before trusting a fold.")
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
    try:
        rel = os.path.relpath(path, REPO)
    except ValueError:
        # Windows raises when the two paths are on different drives, which is
        # every clip on the external drive. There is no relative form; keep it
        # absolute. These strings are only ever compared against each other
        # (load_index intersects the manifest and the index on "path").
        rel = os.path.abspath(path)
    return {
        "path": rel,
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


# Only these Background sources may be pruned by model score. Everything else in
# the class carries a label a person put there, and for those a high score means
# the model is wrong, not the label. Measured on the current model: it flags
# 64.9 % of curated Cercocebus recordings, 60.5 % of curated Pan troglodytes, and
# 72.3 % of the windows a reviewer listened to and judged not a call. Pruning by
# score would delete exactly the hard negatives the model most needs, and would
# do it in proportion to how badly the model is failing on them.
# ``auto_flagged_fp`` itself is raw filter output.  A suffix means that a
# reviewer has resolved the clip (for example ``:confirmed_fp``), and must
# never be deleted merely because a model disagrees with that human verdict.
MACHINE_PRUNABLE_SOURCE = "auto_flagged_fp"


def drop_call_like_negatives(df, scores_csv, threshold=0.5):
    """Remove Background clips the detector thinks are calls, where the label
    is a machine's opinion rather than a person's.

    17 101 of the Background rows are BirdNET's own detections on the deployment
    audio, chosen by a bird detector and never listened to individually. Scored
    with the current model, 98 % of them sit at 0.0000 on every target group, so
    the class is overwhelmingly what it claims to be. A small tail is not:
    275 clips reach 0.5 and some reach 1.000.

    Those are dropped rather than kept. A negative that both a bird detector and
    this detector consider call-like is either a genuine call filed as noise, in
    which case training on it teaches the model that a call is not a call, or a
    hard negative, in which case losing 275 of 17 101 costs almost nothing. The
    asymmetry is the whole argument: one error is unrecoverable and the other is
    rounding.

    They are dropped, not reassigned. Nobody has listened to them, and moving an
    unlistened clip into a positive class would repeat the mistake in the other
    direction. They sit in data/outputs/birdnet_check for a human ear.
    """
    if not os.path.exists(scores_csv):
        print(f"  ! {scores_csv} missing; no call-like negatives dropped")
        return df
    s = pd.read_csv(scores_csv)
    hot = set(s.loc[s.target_score >= threshold, "path"])
    if not hot:
        return df
    source = df["source"].fillna("").astype(str)
    # Be deliberately conservative with a missing or unfamiliar value: only an
    # explicitly false verification status is eligible for model-score pruning.
    # That makes this function safe if it is ever applied before/after review
    # import or to a CSV whose booleans were round-tripped as strings.
    verified_text = df["verified"].fillna("").astype(str).str.strip().str.casefold()
    explicitly_unverified = verified_text.isin({"false", "0", "no", "n"})
    prunable = (
        (source.str.startswith("birdnet:") | source.eq(MACHINE_PRUNABLE_SOURCE))
        & explicitly_unverified
    )
    before = len(df)
    keep = ~(df["label"].eq("Background") & prunable & df["path"].isin(hot))
    dropped = df[~keep]
    df = df[keep].reset_index(drop=True)
    if len(dropped):
        by = dropped["source"].str.replace(r"birdnet:.*", "birdnet(all)",
                                           regex=True).value_counts()
        print(f"  dropped {before - len(df)} Background clips scoring >= "
              f"{threshold} on a target group:")
        for src, n in by.items():
            print(f"    {n:5d}  {src}")
    spared = int((df["label"].eq("Background") & ~prunable
                  & df["path"].isin(hot)).sum())
    if spared:
        print(f"  kept {spared} human-labelled negatives the model also flags; "
              f"on those a high score is the model's error, not the label's")
    return df


def augment_to_target(df, target=3000):
    """Replicate each target class's rows until it holds ``target`` of them.

    Sun et al. (2022) augment to a fixed count per class rather than by a fixed
    multiplier, and report that transfer learning without augmentation is not
    enough on small classes: with augmentation their accuracy went 51.4 to 90.4
    and their F-measure 42.8 to 89.2. This pipeline had no augmentation at all,
    which puts it in the quadrant that study measured as insufficient.

    Augmenting to a count rather than by a multiplier does a second thing worth
    having. The classes here span 150 to 26 323 clips, and `balanced` class
    weights turn that into a 284-fold spread in per-sample weight, which is why
    adding negatives raised every target class's weight instead of lowering it.
    Levelling the target classes at one count compresses that spread to about
    nine-fold without touching the weighting scheme.

    Background is left alone: it is already the largest class, and replicating
    ambient teaches nothing. The rows carry an ``aug`` index; row 0 is the
    original and the packer applies a transformation for the rest. Every variant
    keeps its parent's path, so ``train.source_group`` groups them and no
    augmented copy can land on the far side of a split from its original.
    """
    out = []
    for label, g in df.groupby("label", sort=False):
        if label == "Background" or len(g) >= target:
            out.append(g.assign(aug=0))
            print(f"  {label:18s} {len(g):5d} -> {len(g):5d}  (unchanged)")
            continue
        parts = []
        k = 0
        while sum(len(p) for p in parts) < target:
            parts.append(g.assign(aug=k))
            k += 1
        got = pd.concat(parts, ignore_index=True).head(target)
        out.append(got)
        print(f"  {label:18s} {len(g):5d} -> {len(got):5d}  "
              f"({got.aug.max() + 1} variants each)")
    return pd.concat(out, ignore_index=True)


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
        # Single roar pulses, cut on the envelope and stored raw at 0.26-1.60 s
        # so load_audio_file embeds each one in a fresh background bed, exactly
        # as it does for hacks, keks and pyows. This replaces the 617 fixed 2 s
        # windows that used to stand here. Those were cut at the loudest point
        # of each recording, so where the crop fell relative to the pulse train
        # was arbitrary and two clips of one roar could be half a pulse apart.
        # The species expert, shown isolated pulses, judged one obvious and one
        # borderline and said the isolation "makes clear the shape/pattern we
        # are looking for", which is the case for trying it. Whether it helps is
        # measured against the nine field positive controls, not assumed.
        # The 5 s clips are gone from this list on purpose, and their pulses are
        # in the folder above. Anything longer than the 2 s window skips
        # embed_in_background entirely and is cropped instead, so those 172
        # clips arrived at a roar-to-soundscape ratio of +12.2 dB against the
        # +/-1.0 dB the field-verified clips sit at: the easiest examples in a
        # class that has no hard ones at all. Cernic can carry such clips
        # because 2 535 of its examples are real field detections at real field
        # levels. Colobus has none, so every example being easy is the whole
        # problem.
        # Reverted from 'Colobus guereza bouts'; see the note in
        # config.SPECIES_FOLDERS. Three-pulse windows cost field sensitivity
        # (2/9 -> 1/9) because the field delivers one smeared 0.7-2.5 s event,
        # not a 1.7 Hz train.
        ("species/Colobus guereza pulses", "Colobus_guereza", True),
        # Contiguous 1.8 s windows over the same roaring regions, with no
        # pulse-count requirement. The pulses above give the class its discrete
        # form; these give it the continuous one, in whatever mixture the
        # recordings contain -- 67 % hold a single pulse, 26 % two, the rest
        # more. That distribution is close to what the field-verified clips show
        # (median one event per 3 s) and is the opposite of what the bout
        # experiment enforced (a minimum of two, median three), which is why
        # bouts made the model demand a rhythm the recorders never deliver.
        # Having both forms present means neither is necessary to be a roar.
        ("species/Colobus guereza chunks", "Colobus_guereza", True),
        # The nine field-verified roars, expert-confirmed, from an independent
        # passive-acoustic deployment. Until now they were held out as the only
        # field positive control this class had, and every change to it was
        # scored on them. They are training data from here on, at the user's
        # decision, and the reasoning is sound: nine clips is too small to
        # measure with -- the difference between the two candidate models was a
        # single clip -- while this class is otherwise 100 % archival, which is
        # the defect that has limited it all along. Nine real field roars change
        # what the class is made of; nine test cases were never going to settle
        # anything. What replaces them as the measurement is a person listening
        # to detections, which is a stronger standard, not a weaker one.
        #
        # They are 3 s and therefore cropped to the loudest 2 s rather than
        # embedded, which is correct here and only here: these were recorded by
        # a passive recorder at field distance, so their signal-to-noise is the
        # real thing rather than something that has to be simulated. Their
        # coordinates are a different deployment, so possible_stations returns
        # empty and no fold can leak through them.
        ("species/Colobus guereza field", "Colobus_guereza", True),
        ("species/Colobus_confuser", "Colobus_confuser", True),
        # Cercopithecus pogonias, the congener the expert identified by ear as
        # the source of the daytime Cernic false positives. A trained class but
        # not a detection target; config.DETECTION_GROUPS folds it into
        # Background at detection time, exactly like Colobus_confuser.
        ("species/C_pogonias", "C_pogonias", True),
        ("background/background noise Clips 5sec", "Background", True),
        ("background/Cercocebus torquatus Clips 5s", "Background", True),
        ("background/Pan troglodytes Clips 5sec", "Background", True),
        ("background/wrong classified", "Background", True),
        # Uniform random draws from the deployment. WITHDRAWN pending human
        # review of the screen that labelled them; see the note in
        # config.BACKGROUND_FOLDERS. Re-enable both lines together.
        # ("background/random_forest", "Background", True),
    ]
    # Recursive so a curated folder may be organised into subfolders. Every
    # folder in spec is currently flat, so this changes nothing today.
    for rel, label, verified in spec:
        for p in sorted(glob.glob(os.path.join(DATA, rel, "**", "*.wav"),
                                  recursive=True)):
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
    are thunder and other low-frequency noise.

    The corroboration is that they cluster on one storm, not that they fall at
    night. This docstring used to argue the second: "90.7 % fall between 19:00
    and 05:00, and guereza roars at dawn". Both halves are wrong. The 90.7 %
    came from a partial join that matched 97 of 253 filenames; over all 253 the
    figure is 48.6 %, and over the 307 clips now on disk it is 40.1 % (123 night
    against 184 day), so the majority fall in daylight. And this project's own
    nine adjudicated field roars are at 08:00 (three), 08:30, 10:30, 11:00
    (two), 17:00 and 17:30 -- all daytime, none between 19:00 and 05:00 -- so
    daytime is the roar window by our own material, and a diel argument points
    the other way if it points anywhere. The correction is recorded at
    train_v13_loso.py:821, reject_array_wide_events.py:77 and
    CORRECTIONS_2026-08-01.md:775; this line was the one place it never reached.

    What does hold up is verifiable on the filenames: 230 of the 307 clips fall
    on 2021-02-24 alone, a storm front. A class of detections that is 75 % one
    afternoon is weather, whatever hour it lands at.

    They are tagged ``colobus_field_fp`` rather than merged into the generic
    negatives so the decision stays reversible, because if a genuine roar is
    hiding in here, training it as a negative is the one mistake that cannot be
    undone by adding more data. This docstring also used to promise a
    ``--flag-suspect`` side file of the clips that most resemble a real roar.
    No such option exists -- the string appears nowhere else in the repository --
    so the reversibility rests on the tag and on the clips themselves remaining
    on disk under data/outputs/detected_clips/Colobus_guereza, re-adjudicable at
    any time. Claiming a safety mechanism that was never written is worse than
    having none, because it stops anyone looking for one.

    They enter as ``Colobus_confuser`` rather than ``Background``. These are the
    sounds this model actually mistakes for a guereza roar, recorded in the
    channel where it makes the mistake, which is what that class exists for; the
    reasoning in ``config.SPECIES_FOLDERS`` for giving the confuser its own
    softmax output applies to them exactly. Both labels route to Background at
    detection time through ``config.DETECTION_GROUPS``, so this changes what the
    model is taught, not what the pipeline reports.
    """
    rows = []
    for p in sorted(glob.glob(os.path.join(CLIPS_COLOBUS, "*", "*", "*.wav"))):
        station = os.path.relpath(p, CLIPS_COLOBUS).split(os.sep)[0]
        rows.append(_row(p, "Colobus_confuser", [station], "colobus_field_fp",
                         True))
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
        # Loudly, not as a warning in a wall of output. Without the drive this
        # silently writes a manifest missing 17 101 of its 31 021 clips -- more
        # than half the negative pool -- and every downstream number would be
        # computed on it without anyone noticing which run they were looking at.
        raise SystemExit(
            f"\nThe external drive is not mounted at:\n  {DRIVE}\n\n"
            f"That folder supplies 17 101 BirdNET negatives and the "
            f"coordinate->station map.\nWithout it this would write a manifest "
            f"missing more than half its negatives,\nwhich looks like a "
            f"successful run.\n\n"
            f"Set DRIVE at the top of this script to wherever the drive is "
            f"mounted\n(macOS: /Volumes/<name>; WSL: /mnt/d/<name>), or pass "
            f"--no-birdnet if you\nreally intend to build without them.\n")

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


def drop_cross_label_duplicates(df):
    """Remove clips whose audio appears under a second label.

    A file copied into a curated folder and left in the folder it came from is
    one sound taught as two things, and the filenames differ so nothing upstream
    notices. Found in the current build: two clips present as both
    ``auto_flagged_fp`` Background and ``Colobus_confuser``, under different
    names, identical by content hash.

    Two out of 31 194 is not why this exists. It exists because the failure is
    silent, the cost of checking is one hash per clip, and the loop that
    produces these -- review a detection, copy it into a class folder -- runs
    every time a person adjudicates anything, so the count only goes up.

    The keeper is the row whose source is most specific about provenance:
    a reviewed or curated label beats a machine-flagged one, because the
    machine-flagged copy is the one nobody looked at.
    """
    import hashlib

    rank = {"review": 0, "reference": 1, "colobus_field_fp": 2,
            "auto_flagged_fp": 3, "birdnet": 4}

    def priority(src):
        return rank.get(str(src).split(":")[0], 9)

    digest, missing = {}, 0
    for p in df["path"]:
        ap = os.path.join(REPO, str(p).replace("\\", "/"))
        if not os.path.exists(ap):
            missing += 1
            digest[p] = None
            continue
        h = hashlib.md5()
        with open(ap, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        digest[p] = h.hexdigest()

    d = df.assign(_h=df["path"].map(digest))
    labelled = (d.dropna(subset=["_h"])
                 .groupby("_h")["label"].nunique())
    clashing = set(labelled[labelled > 1].index)
    if not clashing:
        print(f"\nCross-label duplicates: none "
              f"({len(d) - missing} clips hashed)")
        return df

    drop = []
    for h in clashing:
        rows = d[d._h == h]
        keep = rows.assign(_p=rows["source"].map(priority)).sort_values("_p").index[0]
        drop.extend([i for i in rows.index if i != keep])
    kept = d.loc[[i for i in d.index if i not in set(drop)]]
    print(f"\nCross-label duplicates: {len(clashing)} audio files carried more "
          f"than one label; dropped {len(drop)} rows, kept the best-attested "
          f"copy of each")
    for h in list(clashing)[:5]:
        rows = d[d._h == h]
        print(f"    {' / '.join(sorted(set(rows.label)))}: "
              f"{os.path.basename(str(rows.path.iloc[0]))}")
    return kept.drop(columns="_h").reset_index(drop=True)


def drop_excluded(df, csv_path):
    """Drop clips a person has ruled out, listed by filename in a CSV.

    Kept separate from the collectors because the reasons are human decisions
    that arrive after the fact and have to be auditable: the CSV carries a
    `reason` and a `decided_by` per clip, and lives in data/labels/ with the
    other irreproducible human labels rather than being hard-coded here.
    """
    if not csv_path or not os.path.exists(csv_path):
        return df, 0
    ex = pd.read_csv(csv_path)
    names = set(ex["file"].astype(str))
    hit = df["path"].map(lambda p: os.path.basename(str(p)) in names)
    if hit.any():
        by = ex.groupby("reason").size().to_dict() if "reason" in ex else {}
        for reason, n in by.items():
            print(f"    {n:5d} listed: {reason}")
    return df[~hit].reset_index(drop=True), int(hit.sum())


def drop_sources(df, names):
    """Drop whole provenance groups, matching the source name EXACTLY.

    Exact and not a prefix, which was the first version and was wrong in a way
    worth recording. Sources here are colon-separated refinements of a common
    stem: `auto_flagged_fp` is the block nobody listened to, but
    `auto_flagged_fp:confirmed_fp` is 317 clips a person confirmed and
    `auto_flagged_fp:RECOVERED_CALL` is 70 clips a person confirmed are real
    C. nictitans calls. A prefix match on "auto_flagged_fp" swept up all three
    and deleted verified positives -- the exact opposite of the intent, and
    invisible except as a class count two lower than expected.

    So dropping a stem never touches its refinements; list them separately if
    that is really what you want.
    """
    if not names:
        return df, 0
    src = df["source"].astype(str)
    hit = pd.Series(False, index=df.index)
    for p in names:
        m = src.eq(p)
        if m.any():
            print(f"    {int(m.sum()):5d} rows from source '{p}'")
        else:
            print(f"    ! nothing matched source '{p}' exactly; "
                  f"related sources present: "
                  f"{sorted(s for s in src.unique() if str(s).startswith(p))}")
        hit |= m
    return df[~hit].reset_index(drop=True), int(hit.sum())


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(DATA, "outputs/v13_manifest.csv"))
    ap.add_argument("--no-birdnet", action="store_true",
                    help="skip the external-drive bird segments")
    ap.add_argument("--exclude", default="", metavar="CSV",
                    help="CSV with a 'file' column naming clips to keep out of "
                         "training, plus 'reason' and 'decided_by'. Applied "
                         "before augmentation, so an excluded clip is never "
                         "replicated.")
    ap.add_argument("--drop-source", action="append", default=[], metavar="PREFIX",
                    help="Drop every row whose source starts with PREFIX. "
                         "Repeatable. For blocks no person has listened to, "
                         "where the honest default is to leave them out rather "
                         "than train on an assumption.")
    ap.add_argument("--drop-call-like", type=float, default=0.0, metavar="T",
                    help="Drop Background clips scoring at or above T on any "
                         "target group, using data/outputs/birdnet_scores.csv. "
                         "0 disables. A negative the detector itself calls a "
                         "call is either a mislabelled call, which is an "
                         "unrecoverable training error, or a hard negative, "
                         "which is cheap to lose.")
    ap.add_argument("--augment-to", type=int, default=0, metavar="N",
                    help="Replicate each target class up to N rows, marking the "
                         "copies with an 'aug' index the packer turns into an "
                         "actual transformation. 0 disables it, which is what "
                         "this pipeline did until now: every clip entered "
                         "training exactly once and unaltered, which is the "
                         "condition Sun et al. (2022) measure as insufficient.")
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
    if args.drop_call_like:
        # background_scores.csv covers every Background source, not just the
        # BirdNET segments: the instruction is that no target call ends up in
        # the negative class, and the curated and auto-mined sources have to
        # clear the same bar as the bird detector's output.
        for f in ("outputs/background_scores.csv", "outputs/birdnet_scores.csv"):
            manifest = drop_call_like_negatives(
                manifest, os.path.join(DATA, f), args.drop_call_like)
    manifest = drop_cross_label_duplicates(manifest)

    # Both of these run before augmentation on purpose: replicating a clip
    # twenty times and then removing it wastes the work, and removing it after
    # a partial replication would leave an inconsistent set.
    if args.drop_source:
        print("\nDropping sources nobody has listened to:")
        manifest, n = drop_sources(manifest, args.drop_source)
        print(f"  dropped {n} rows")
    if args.exclude:
        print("\nApplying the expert exclusion list:")
        manifest, n = drop_excluded(manifest, args.exclude)
        print(f"  dropped {n} rows")

    if args.augment_to:
        print(f"\nAugmenting the target classes to {args.augment_to} rows each")
        manifest = augment_to_target(manifest, args.augment_to)
        manifest = manifest.reset_index(drop=True)
    else:
        manifest = manifest.assign(aug=0)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    manifest.to_csv(args.out, index=False)
    report(manifest)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
