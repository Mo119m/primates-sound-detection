"""
Automatic false-positive cleanup for detection results.

Runs three independent filters over saved detection CSVs and splits the
detections into "clean" (trustworthy without listening) and "suspicious":

1. Mahalanobis OOD   - feature distance to the predicted class's training
   cluster. A real call lives close to its training cluster; a bird call does
   not.
2. YAMNet cross-check - Google's 521-class audio tagger flags windows whose top
   class is Bird/Insect/Wind/Rain/Speech/etc. OFF by default: measured against
   a manual review of 6189 field detections it flagged more genuine calls than
   false positives (see config.USE_YAMNET_FILTER for the numbers).
3. Temporal isolation - primates call in bouts, so a detection with no
   same-species neighbour within +/- isolation_window_s is suspicious.
4. Station regime  - a station overrun by an untrained species produces a dense
   cluster that none of the above can see, being neither isolated nor outlying.
   Each station is classified from its own detections and the cluster removed
   only where one is found (see src/station_regime.py).

A detection is clean only if every enabled filter passes it. Hard negatives for
the next retraining iteration are the detections flagged by >= 2 filters, plus
any invading cluster, which the other filters cannot corroborate.

Use :func:`run_auto_cleanup` as the single entry point, or call the individual
``filter_*`` helpers for finer control.
"""

import os
import re
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import soundfile as sf

try:
    from . import config
    from . import data_loader
    from . import preprocessing
    from . import model as model_module
    from . import station_regime
    from . import review_queue
except ImportError:  # Allow running as a standalone script (e.g. in Colab)
    import config
    import data_loader
    import preprocessing
    import model as model_module
    import station_regime
    import review_queue


# AudioSet classes that indicate the window is NOT one of our primates.
# "Animal" and "Wild animals" are kept as plausible primate labels.
DEFAULT_SUSPICIOUS_YAMNET = {
    'Bird', 'Bird vocalization, bird call, bird song', 'Chirp, tweet',
    'Squawk', 'Pigeon, dove', 'Crow', 'Owl', 'Gull, seagull',
    'Insect', 'Cricket', 'Cicada', 'Mosquito', 'Fly, housefly', 'Bee, wasp, etc.',
    'Frog', 'Snake', 'Rattle',
    'Wind', 'Wind noise (microphone)', 'Rustling leaves', 'Rain', 'Rain on surface',
    'Thunder', 'Thunderstorm', 'Stream', 'Waterfall',
    'Silence', 'Speech', 'Male speech, man speaking', 'Female speech, woman speaking',
    'Conversation', 'Narration, monologue', 'Static', 'White noise', 'Pink noise',
    'Hum', 'Buzz', 'Mains hum',
    'Music', 'Fireworks', 'Explosion', 'Glass', 'Noise',
    'Bell', 'Bicycle bell', 'Bicycle', 'Buzzer', 'Ratchet, pawl',
}


# FEATURE EXTRACTION + MAHALANOBIS STATISTICS

def build_feature_extractor(model, layer_name: str = 'dense_256',
                            with_probs: bool = False):
    """Tap an intermediate dense layer to get a per-window feature vector.

    With ``with_probs`` the model also returns the class probabilities, so the
    feature vector and the softmax come from a single forward pass.
    """
    import tensorflow as tf
    feat_layer = model.get_layer(layer_name)
    outputs = [feat_layer.output, model.output] if with_probs else feat_layer.output
    return tf.keras.Model(inputs=model.inputs, outputs=outputs)


def clips_to_model_input(clips):
    """Stack detection clips into the batch the model expects."""
    return np.stack([_audio_to_model_input(c, config.SAMPLE_RATE)
                     for c in clips]).astype(np.float32)


def _audio_to_model_input(audio, sr):
    img = preprocessing.preprocess_audio(audio, sr)
    return preprocessing.preprocess_for_model(img)


def _mahalanobis(features, class_idx, class_means, inv_cov):
    diff = features - class_means[class_idx]
    return np.einsum('bi,ij,bj->b', diff, inv_cov, diff)


def _group_members():
    """Map each coarse detection label to the training-class indices it covers."""
    members = {}
    for cls_name, group in config.DETECTION_GROUPS.items():
        members.setdefault(group, []).append(config.CLASS_NAMES.index(cls_name))
    return members


def compute_class_statistics(feature_extractor, species_data, background_data,
                             cache_path=None, ridge: float = 1e-4, verbose: bool = True):
    """
    Compute per-class feature means and a pooled inverse covariance over the
    original (non-augmented) training clips. Cached to ``cache_path`` (.npz).

    Returns:
        (class_means, inv_cov, train_feats_by_class)
    """
    if cache_path and os.path.exists(cache_path):
        if verbose:
            print(f'Loading cached class stats from {cache_path}')
        z = np.load(cache_path)
        class_means = z['class_means']
        inv_cov = z['inv_cov']
        train_feats_by_class = {int(k): z[f'feats_{int(k)}'] for k in z['class_ids']}
        return class_means, inv_cov, train_feats_by_class

    label_map = {name: i for i, name in enumerate(config.CLASS_NAMES)}
    X_list, y_list = [], []
    for sp_name, audio_list in species_data.items():
        for audio, _ in audio_list:
            X_list.append(_audio_to_model_input(audio, config.SAMPLE_RATE))
            y_list.append(label_map[sp_name])
    for audio, _ in background_data:
        X_list.append(_audio_to_model_input(audio, config.SAMPLE_RATE))
        y_list.append(label_map['Background'])
    X_arr = np.array(X_list, dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.int64)
    if verbose:
        print(f'Extracting features for {len(X_arr)} training clips...')

    feats = feature_extractor.predict(X_arr, batch_size=config.BATCH_SIZE,
                                      verbose=1 if verbose else 0)

    class_ids = sorted(np.unique(y_arr).tolist())
    class_means = np.zeros((len(config.CLASS_NAMES), feats.shape[1]), dtype=np.float32)
    train_feats_by_class = {}
    centered_parts = []
    for c in class_ids:
        fc = feats[y_arr == c]
        mu = fc.mean(axis=0)
        class_means[c] = mu
        train_feats_by_class[c] = fc
        centered_parts.append(fc - mu)
    centered = np.concatenate(centered_parts, axis=0)
    cov = np.cov(centered, rowvar=False) + ridge * np.eye(centered.shape[1], dtype=np.float32)
    inv_cov = np.linalg.inv(cov).astype(np.float32)

    if cache_path:
        save_kwargs = {'class_means': class_means, 'inv_cov': inv_cov,
                       'class_ids': np.array(class_ids)}
        for c, fc in train_feats_by_class.items():
            save_kwargs[f'feats_{c}'] = fc
        np.savez(cache_path, **save_kwargs)
        if verbose:
            print(f'Saved class stats to {cache_path}')

    return class_means, inv_cov, train_feats_by_class


def calibrate_thresholds(train_feats_by_class, class_means, inv_cov,
                         percentile: int = 95):
    """Per-class Mahalanobis cutoff = ``percentile`` of in-distribution distances."""
    thresholds = {}
    for c, fc in train_feats_by_class.items():
        d2 = _mahalanobis(fc, c, class_means, inv_cov)
        thresholds[c] = float(np.percentile(d2, percentile))
    return thresholds


# ADDITIONAL PER-DETECTION SIGNALS
# Both are recorded as continuous columns rather than applied as flags, so
# their usefulness can be measured against a manual review (see
# cleanup_eval.signal_sweep) before any cutoff is trusted.

def annotate_softmax_margin(det_df: pd.DataFrame, probs) -> pd.DataFrame:
    """
    Add the gap between the top two class probabilities.

    The reported confidence is the top probability alone, which saturates near
    1 for almost every detection and so separates little. How far the winning
    class sits above the runner-up keeps varying where the top probability no
    longer does, and a window the model finds genuinely ambiguous shows a small
    margin even when its top score is high.
    """
    det_df = det_df.copy()
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2 or len(p) != len(det_df):
        det_df['softmax_margin'] = np.nan
        det_df['softmax_entropy'] = np.nan
        return det_df
    part = np.partition(p, -2, axis=1)
    det_df['softmax_margin'] = part[:, -1] - part[:, -2]
    with np.errstate(divide='ignore', invalid='ignore'):
        det_df['softmax_entropy'] = -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=1)
    return det_df


def annotate_station_recurrence(det_df: pd.DataFrame, feats, k: int = 5,
                                group_cols=('station', 'species'),
                                verbose: bool = True) -> pd.DataFrame:
    """
    Measure how repetitive each detection is within its station.

    A recurring non-target sound -- one insect chorus, one machine, one bird
    calling all night at a single station -- produces many detections whose
    feature vectors are near copies of each other. The Mahalanobis filter
    cannot see this: once such a sound is numerous it stops being an outlier
    and starts defining the distribution it would be measured against. Genuine
    calls vary between utterances, so they sit further from their neighbours.

    ``recurrence_knn_dist`` is the distance to the ``k``-th nearest detection
    of the same species at the same station. Small means the detection is one
    of many near-identical copies. A k-th neighbour distance is used rather
    than a count inside a fixed radius because any such radius would have to be
    chosen from the same distances it is meant to judge. It is a measurement,
    not a verdict: sweep it against a manual review before thresholding it.
    """
    det_df = det_df.copy()
    feats = np.asarray(feats, dtype=np.float32)
    if len(feats) != len(det_df):
        det_df['recurrence_knn_dist'] = np.nan
        return det_df

    cols = [c for c in group_cols if c in det_df.columns]
    if cols:
        keys = det_df[cols].astype(str).agg('||'.join, axis=1).to_numpy()
    else:
        keys = np.zeros(len(det_df), dtype=int)
    out = station_regime.knn_distance(feats, keys, k=k)

    det_df['recurrence_knn_dist'] = out
    if verbose and np.isfinite(out).any():
        finite = out[np.isfinite(out)]
        print(f'  Recurrence: {k}-NN distance median {np.median(finite):.2f}, '
              f'min {finite.min():.2f} (small = repetitive)')
    return det_df


def filter_station_regime(det_df: pd.DataFrame,
                          min_frac=station_regime.DEFAULT_MIN_CLUSTER_FRACTION,
                          min_gap_ratio=station_regime.DEFAULT_MIN_GAP_RATIO,
                          verbose: bool = True) -> pd.DataFrame:
    """
    Decide per station whether it has been overrun, and flag the cluster if so.

    Stations fail in two ways that need different treatment. Scattered false
    positives respond to the temporal-isolation rule, which is mild enough to
    run everywhere. A station taken over by one untrained species does not:
    those detections are numerous and consistent, so they are neither isolated
    nor outliers, and only removing the cluster wholesale reaches them --
    which would be far too destructive at a station with no invasion.

    The decision is made without labels and without any cross-station
    threshold; see ``station_regime`` for why both matter. Adds
    ``station_regime`` ('invaded' or 'normal') and ``flag_invading_cluster``.
    """
    det_df = det_df.copy()
    if 'recurrence_knn_dist' not in det_df.columns:
        det_df['station_regime'] = station_regime.NORMAL
        det_df['flag_invading_cluster'] = False
        return det_df

    # A flat layout leaves 'station' blank for every row, which would group all
    # recordings into a single pseudo-station; fall back to the recording.
    group_col = 'station'
    if ('station' not in det_df.columns
            or det_df['station'].astype(str).str.strip().eq('').all()):
        group_col = 'source_file'

    mask = station_regime.detect_invading_cluster(
        det_df, group_col=group_col,
        min_frac=min_frac, min_gap_ratio=min_gap_ratio)
    det_df['flag_invading_cluster'] = mask.to_numpy()
    det_df['station_regime'] = station_regime.classify_stations(
        det_df, mask=mask, group_col=group_col).to_numpy()

    if verbose:
        n = int(mask.sum())
        sites = sorted(det_df.loc[mask, group_col].unique()) if n else []
        if n:
            print(f'  Station regime: {len(sites)} station(s) look invaded '
                  f'({", ".join(map(str, sites))}); flagged {n} detections in '
                  f'their dominant cluster.')
        else:
            print('  Station regime: no station shows a dominant unfamiliar '
                  'cluster; the cluster rule stays off.')
    return det_df


# DETECTION LOADING + CLIP EXTRACTION

def load_detection_csvs(detection_dir=None) -> pd.DataFrame:
    """
    Load every ``*_detections.csv`` under ``detection_dir`` (recursively, so
    per-station subfolders are included) and attach the resolved source-audio
    path for each detection.
    """
    detection_dir = Path(os.path.expanduser(
        str(detection_dir or config.DETECTION_OUTPUT_DIR)))
    csv_files = sorted(detection_dir.rglob('*_detections.csv'))
    if not csv_files:
        raise FileNotFoundError(f'No detection CSVs under {detection_dir}')

    audio_index = {os.path.basename(p): p for p in data_loader.get_long_audio_files()}
    if os.path.isdir(config.IPA_ROOT):
        for p in data_loader.get_long_audio_files(root=config.IPA_ROOT):
            audio_index.setdefault(os.path.basename(p), p)

    rows = []
    for csv in csv_files:
        df = pd.read_csv(csv)
        if len(df) == 0:
            continue
        source_name = csv.stem.replace('_detections', '') + '.wav'
        df['source_file'] = source_name
        df['source_path'] = audio_index.get(source_name, '')
        # Recordings are stored one folder per station, so the CSV's parent
        # relative to the scan root names the station ('' for a flat layout).
        rel = csv.parent.relative_to(detection_dir)
        df['station'] = str(rel) if str(rel) != '.' else ''
        rows.append(df)
    if not rows:
        raise ValueError('All detection CSVs were empty - nothing to clean up.')

    det_df = pd.concat(rows, ignore_index=True)
    det_df['det_id'] = np.arange(len(det_df))
    return det_df


# <species>__<recording>__<start>s__conf<confidence>.wav, as written by
# utils.extract_all_detected_clips.
_EXPORTED_CLIP_RE = re.compile(
    r"^(?P<species>.+?)__(?P<recording>.+)__(?P<start>\d+)s__conf[0-9.]+\.wav$")


def load_clips_from_dir(det_df: pd.DataFrame, clips_dir, padding: float = 0.5,
                        start_tolerance: int = 1, verbose: bool = True):
    """
    Take each detection's audio from the clips already exported for manual
    review, instead of re-cutting it from the source recording.

    ``utils.extract_all_detected_clips`` writes one WAV per detection, padded by
    ``padding`` seconds on each side, so the analysis window can be recovered
    from the clip alone -- the long recordings are not needed. Clips are matched
    to detections on ``(species, recording, start second)``.

    Args:
        det_df: detection table (needs species, source_file, start_time).
        clips_dir: folder written by extract_all_detected_clips.
        padding: seconds of padding used at export (its default is 0.5).
        start_tolerance: seconds of slack when matching a clip to a detection.
    """
    clip_len = int(round(config.WINDOW_SIZE * config.SAMPLE_RATE))

    # Expand '~' explicitly: the shell does it, Python does not, so a pasted
    # '~/...' path would otherwise match nothing.
    clips_dir = os.path.expanduser(str(clips_dir))

    index = {}
    for path in glob.glob(os.path.join(clips_dir, "**", "*.wav"),
                          recursive=True):
        m = _EXPORTED_CLIP_RE.match(os.path.basename(path))
        if m:
            index.setdefault((m.group("species"), m.group("recording"),
                              int(m.group("start"))), path)
    if verbose:
        print(f"  indexed {len(index)} exported clips under {clips_dir}")

    clips = []
    missing_rows = 0
    missing_files = set()
    for row in det_df.itertuples():
        recording = os.path.splitext(str(getattr(row, "source_file", "")))[0]
        start_t = float(row.start_time)
        base = (str(row.species), recording, int(round(start_t)))
        path = None
        for delta in [0] + [d for t in range(1, int(start_tolerance) + 1)
                            for d in (-t, t)]:
            path = index.get((base[0], base[1], base[2] + delta))
            if path:
                break

        if not path:
            missing_rows += 1
            missing_files.add(f"{base[0]}__{base[1]}__{base[2]}s")
            clips.append(np.zeros(clip_len, dtype=np.float32))
            continue

        y, _ = librosa.load(path, sr=config.SAMPLE_RATE, mono=True)
        # The exporter cut from max(0, start - padding), so the analysis window
        # begins min(start, padding) seconds into the clip.
        offset = int(round(min(start_t, padding) * config.SAMPLE_RATE))
        clip = y[offset:offset + clip_len]
        if len(clip) < clip_len:
            clip = np.pad(clip, (0, clip_len - len(clip)))
        clips.append(clip)

    if missing_rows:
        examples = ", ".join(sorted(missing_files)[:3])
        if missing_rows == len(det_df):
            raise FileNotFoundError(
                f"None of the {len(det_df)} detections matched an exported clip "
                f"under {clips_dir!r}, so every clip would be silence. Check "
                f"that this folder holds the clips for these detections. "
                f"Missing e.g.: {examples}")
        if verbose:
            print(f"  WARNING: {missing_rows} of {len(det_df)} detections had no "
                  f"exported clip and were replaced with silence; their verdicts "
                  f"are not trustworthy. Missing e.g.: {examples}")
    return clips


def extract_clips(det_df: pd.DataFrame, verbose: bool = True):
    """Cut one ``WINDOW_SIZE`` clip per detection, caching each source file.

    A detection whose source recording cannot be found yields a silent clip.
    That silently corrupts every filter downstream, so the misses are counted
    and reported, and an all-missing run is treated as a hard error.
    """
    clip_len = int(round(config.WINDOW_SIZE * config.SAMPLE_RATE))
    cache = {}
    clips = []
    missing_rows = 0
    missing_files = set()
    for row in det_df.itertuples():
        path = row.source_path
        if not path or not os.path.exists(path):
            missing_rows += 1
            missing_files.add(getattr(row, 'source_file', path) or '(unknown)')
            clips.append(np.zeros(clip_len, dtype=np.float32))
            continue
        if path not in cache:
            cache[path], _ = librosa.load(path, sr=config.SAMPLE_RATE, mono=True)
        y = cache[path]
        s = int(round(row.start_time * config.SAMPLE_RATE))
        clip = y[s:s + clip_len]
        if len(clip) < clip_len:
            clip = np.pad(clip, (0, clip_len - len(clip)))
        clips.append(clip)

    if missing_rows:
        examples = ', '.join(sorted(missing_files)[:3])
        if missing_rows == len(det_df):
            raise FileNotFoundError(
                f'None of the {len(det_df)} detections could be matched to a '
                f'source recording, so every clip would be silence and the '
                f'cleanup verdicts would be meaningless. Point '
                f'LONG_AUDIO_ROOT (currently {config.LONG_AUDIO_ROOT!r}) at the '
                f'recordings these detections came from. Missing e.g.: {examples}')
        if verbose:
            print(f'  WARNING: {missing_rows} of {len(det_df)} detections have '
                  f'no source recording and were replaced with silence; their '
                  f'verdicts are not trustworthy. Missing e.g.: {examples}')
    return clips


# THE THREE FILTERS

def filter_mahalanobis(det_df, clips, feature_extractor, class_means, inv_cov,
                       class_thresholds, percentile: int = 95,
                       calibrate_on: str = 'detections',
                       feats=None, verbose: bool = True) -> pd.DataFrame:
    """Flag detections whose feature vector is OOD.

    calibrate_on='detections' (default) computes per-species thresholds from
    the detection distances themselves, avoiding domain-shift false alarms
    when noisy field recordings differ from clean training clips.
    calibrate_on='training' uses the training-data thresholds (class_thresholds).
    """
    if feats is None:
        X = clips_to_model_input(clips)
        feats = feature_extractor.predict(X, batch_size=config.BATCH_SIZE,
                                          verbose=1 if verbose else 0)
    members_map = _group_members()

    def members_for(label):
        if label in members_map:
            return members_map[label]
        return [config.CLASS_NAMES.index(label)]

    scores = np.zeros(len(det_df), dtype=np.float32)
    species = det_df['species'].to_numpy()

    if calibrate_on == 'detections':
        for i in range(len(det_df)):
            members = members_for(species[i])
            d2s = [float(_mahalanobis(feats[i:i + 1], c, class_means, inv_cov)[0])
                   for c in members]
            scores[i] = min(d2s)
        flags = np.zeros(len(det_df), dtype=bool)
        for sp in np.unique(species):
            mask = species == sp
            thresh = float(np.percentile(scores[mask], percentile))
            flags[mask] = scores[mask] > thresh
            if verbose:
                print(f'  Mahalanobis {sp}: threshold={thresh:.1f}, '
                      f'flagged {int(flags[mask].sum())}/{int(mask.sum())}')
    else:
        flags = np.zeros(len(det_df), dtype=bool)
        for i in range(len(det_df)):
            members = members_for(species[i])
            d2s = [float(_mahalanobis(feats[i:i + 1], c, class_means, inv_cov)[0])
                   for c in members]
            scores[i] = min(d2s)
            flags[i] = all(d2s[k] > class_thresholds[members[k]]
                           for k in range(len(members)))

    det_df = det_df.copy()
    det_df['mahalanobis_d2'] = scores
    # The distance is always kept: it is a real if modest ranking signal (ROC
    # area 0.763 against the 6 189 human verdicts) and review_ranking uses it.
    # The *flag* is what measures as chance (0.485) and what fed genuine calls
    # into Background through the Step-5 loop, so it is off by default -- see
    # config.USE_MAHALANOBIS_FILTER.
    if getattr(config, 'USE_MAHALANOBIS_FILTER', True):
        det_df['flag_mahal'] = flags
        if verbose:
            print(f'Mahalanobis flagged {int(flags.sum())} / {len(det_df)} detections')
    else:
        det_df['flag_mahal'] = False
        if verbose:
            print(f'  [off] Mahalanobis FILTER disabled '
                  f'(config.USE_MAHALANOBIS_FILTER); it would have flagged '
                  f'{int(flags.sum())}. The distance is still computed and '
                  f'still used for ranking.')
    return det_df


def filter_yamnet(det_df, clips, suspicious=None, verbose: bool = True) -> pd.DataFrame:
    """Flag detections whose top YAMNet class is a known non-primate sound.

    YAMNet needs ``tensorflow_hub`` (which imports ``pkg_resources`` from
    setuptools) and a one-time model download from the internet. If either is
    unavailable, this filter is skipped gracefully — every detection is left
    unflagged and the other two filters still run — instead of crashing the
    whole cleanup. Install ``setuptools<81`` (see SETUP.md) and reconnect to
    re-enable it.
    """
    det_df = det_df.copy()
    try:
        import tensorflow as tf
        import tensorflow_hub as hub

        yamnet = hub.load('https://tfhub.dev/google/yamnet/1')
        class_map_path = yamnet.class_map_path().numpy().decode('utf-8')
        yam_classes = pd.read_csv(class_map_path)['display_name'].tolist()
    except Exception as exc:  # noqa: BLE001 - import, download, or TF errors
        print(f'  [skip] YAMNet filter unavailable ({type(exc).__name__}: {exc}).')
        print('         Running without it — the other two filters still apply.')
        print('         To enable it: pip install "setuptools<81" and reconnect '
              '(see SETUP.md).')
        det_df['yamnet_top'] = '(yamnet skipped)'
        det_df['yamnet_score'] = 0.0
        det_df['flag_yamnet'] = False
        return det_df

    suspicious = suspicious or DEFAULT_SUSPICIOUS_YAMNET

    top_class, top_score = [], []
    flags = np.zeros(len(det_df), dtype=bool)
    for i, clip in enumerate(clips):
        clip16 = librosa.resample(clip.astype(np.float32),
                                  orig_sr=config.SAMPLE_RATE, target_sr=16000)
        scores, _, _ = yamnet(clip16)
        mean_scores = tf.reduce_mean(scores, axis=0).numpy()
        j = int(np.argmax(mean_scores))
        top_class.append(yam_classes[j])
        top_score.append(float(mean_scores[j]))
        flags[i] = yam_classes[j] in suspicious
        if verbose and (i + 1) % 50 == 0:
            print(f'  YAMNet {i + 1}/{len(clips)}')

    det_df = det_df.copy()
    det_df['yamnet_top'] = top_class
    det_df['yamnet_score'] = top_score
    det_df['flag_yamnet'] = flags
    if verbose:
        print(f'YAMNet flagged {int(flags.sum())} / {len(det_df)} detections')
    return det_df


def filter_temporal_isolation(det_df, window_s: float = 30.0,
                              min_neighbours: int = None,
                              verbose: bool = True) -> pd.DataFrame:
    """
    Flag detections with fewer than ``min_neighbours`` same-species neighbours
    within +/- ``window_s``.

    ``min_neighbours`` defaults to ``config.ISOLATION_MIN_NEIGHBOURS``. It used
    to be fixed at 1, i.e. "flag only detections with no neighbour at all",
    which discards everything the count knows: graded and applied after the
    time gate the same number reaches ROC area 0.880 against the human verdicts,
    beating the detector's own confidence. See the comment on
    ``ISOLATION_MIN_NEIGHBOURS`` in config for the calibration and for why the
    time gate has to come first.
    """
    if min_neighbours is None:
        min_neighbours = getattr(config, 'ISOLATION_MIN_NEIGHBOURS', 1)
    det_df = det_df.copy()
    iso = np.zeros(len(det_df), dtype=bool)
    n_neigh = np.zeros(len(det_df), dtype=int)
    pos = {det_id: i for i, det_id in enumerate(det_df['det_id'].to_numpy())}
    for _, grp in det_df.groupby(['source_file', 'species']):
        starts = grp['start_time'].to_numpy()
        ids = grp['det_id'].to_numpy()
        for k in range(len(starts)):
            diffs = np.abs(starts - starts[k])
            diffs[k] = np.inf
            n = int((diffs <= window_s).sum())
            n_neigh[pos[ids[k]]] = n
            iso[pos[ids[k]]] = (n < min_neighbours)
    det_df['n_neighbours'] = n_neigh
    det_df['flag_isolated'] = iso
    if verbose:
        print(f'Temporal-isolation flagged {int(iso.sum())} / {len(det_df)} '
              f'detections (n_neighbours < {min_neighbours})')
    return det_df


# MERGE + SAVE

def merge_flags(det_df: pd.DataFrame) -> pd.DataFrame:
    """Add ``n_flags`` and a human-readable ``flag_reason`` column."""
    det_df = det_df.copy()
    flag_cols = ['flag_mahal', 'flag_yamnet', 'flag_isolated',
                 'flag_invading_cluster']
    for c in flag_cols:
        if c not in det_df.columns:
            det_df[c] = False
    det_df['n_flags'] = det_df[flag_cols].sum(axis=1).astype(int)

    def reason(row):
        parts = []
        if row.flag_mahal:
            parts.append('mahal')
        if row.flag_yamnet:
            parts.append(f'yamnet:{row.yamnet_top}')
        if row.flag_isolated:
            parts.append('isolated')
        if getattr(row, 'flag_invading_cluster', False):
            parts.append('invading_cluster')
        return '|'.join(parts)

    det_df['flag_reason'] = det_df.apply(reason, axis=1)
    return det_df


def summarize(det_df: pd.DataFrame) -> pd.DataFrame:
    """Per-species clean/suspicious/strong-FP counts."""
    return det_df.groupby('species').agg(
        total=('det_id', 'count'),
        clean=('n_flags', lambda x: int((x == 0).sum())),
        suspicious=('n_flags', lambda x: int((x > 0).sum())),
        strong_fp=('n_flags', lambda x: int((x >= 2).sum())),
        invaded=('flag_invading_cluster', 'sum')
        if 'flag_invading_cluster' in det_df.columns else ('det_id', 'size'),
    )


def save_hard_negatives(strong_fp_df, clips, fp_dir):
    """Save >=2-flag clips as WAVs under ``fp_dir/<primary_reason>/``."""
    fp_dir = Path(fp_dir)

    def primary_reason(row):
        if row.flag_mahal:
            return 'mahal'
        if row.flag_yamnet:
            return 'yamnet'
        return 'isolated'

    n_saved = 0
    for row in strong_fp_df.itertuples():
        clip = clips[row.det_id]
        sub = fp_dir / primary_reason(row)
        sub.mkdir(parents=True, exist_ok=True)
        stem = os.path.splitext(row.source_file)[0]
        fname = (f'{row.species}__{stem}__t{int(row.start_time):05d}s'
                 f'__conf{row.confidence:.2f}.wav')
        sf.write(sub / fname, clip, config.SAMPLE_RATE)
        n_saved += 1
    return n_saved


def save_clips_by_species(det_df, clips, out_dir):
    """Save each detection's 2 s clip under ``out_dir/<species>/`` so they can
    be reviewed after the original long recording has been deleted."""
    out_dir = Path(out_dir)
    n_saved = 0
    for row in det_df.itertuples():
        clip = clips[row.det_id]
        sub = out_dir / row.species
        sub.mkdir(parents=True, exist_ok=True)
        stem = os.path.splitext(row.source_file)[0]
        fname = (f'{stem}__t{int(row.start_time):05d}s'
                 f'__conf{row.confidence:.2f}.wav')
        sf.write(sub / fname, clip, config.SAMPLE_RATE)
        n_saved += 1
    return n_saved


# ORCHESTRATOR

def run_auto_cleanup(model=None, model_path=None, detection_dir=None,
                     clips_dir=None, clips_padding: float = 0.5,
                     output_dir=None, fp_dir=None,
                     species_data=None, background_data=None,
                     percentile: int = 95, isolation_window_s: float = 30.0,
                     suspicious_yamnet=None, use_yamnet=None,
                     save_clips: bool = True,
                     save_all_clips: bool = False,
                     use_cached_stats: bool = True,
                     mahal_calibration: str = 'detections',
                     verbose: bool = True) -> dict:
    """
    Run the full three-filter cleanup over saved detection CSVs.

    Args:
        model: a loaded Keras model. If None, loaded from ``model_path``.
        model_path: path to best_model.h5 (default: config.MODEL_SAVE_DIR).
        detection_dir: dir holding *_detections.csv (default:
            config.DETECTION_OUTPUT_DIR; searched recursively).
        clips_dir: optional folder of clips already exported by
            ``utils.extract_all_detected_clips``. When given, each detection's
            audio is read from its exported clip instead of being re-cut from
            the source recording, so the long recordings are not needed.
        clips_padding: seconds of padding used when those clips were exported
            (extract_all_detected_clips defaults to 0.5).
        output_dir: where to write per-run results (clean/suspicious CSVs).
            Default: config.OUTPUT_ROOT/auto_cleanup. Pass a per-station path
            when processing one IPA station at a time.
        fp_dir: where strong-FP clips are saved as hard negatives. Defaults to
            the *global* config.OUTPUT_ROOT/auto_cleanup/auto_flagged_fp so
            negatives accumulate in a single pool across stations — this is
            the folder referenced by BACKGROUND_FOLDERS, so all per-station
            FPs feed the next training round.
        species_data, background_data: pre-loaded training data; loaded via
            data_loader if omitted.
        percentile: percentile for the Mahalanobis cutoff (applied to either
            training or detection distances depending on mahal_calibration).
        isolation_window_s: temporal-isolation neighbour window in seconds.
        suspicious_yamnet: set of AudioSet class names to treat as non-primate.
        use_yamnet: run the YAMNet cross-check. Defaults to
            config.USE_YAMNET_FILTER, which is off -- see the note there for the
            measured reason.
        save_clips: write >=2-flag clips as hard negatives.
        save_all_clips: also write every clean and suspicious clip under
            output_dir/clean_clips/<species>/ and suspicious_clips/<species>/
            so they can be reviewed after the long recording is deleted.
            Recommended for the per-station upload-process-delete workflow.
        use_cached_stats: reuse a cached class_stats.npz if present.
        mahal_calibration: 'detections' (default) calibrates Mahalanobis
            thresholds on field-detection distances to avoid domain-shift
            false alarms; 'training' uses clean training-data thresholds.

    Returns:
        dict with keys: det_df, clean_df, suspicious_df, strong_fp_df, summary,
        class_thresholds, output_dir.
    """
    output_dir = Path(output_dir or (Path(config.OUTPUT_ROOT) / 'auto_cleanup'))
    output_dir.mkdir(parents=True, exist_ok=True)
    fp_dir = Path(fp_dir or (Path(config.OUTPUT_ROOT) / 'auto_cleanup' / 'auto_flagged_fp'))

    if model is None:
        model_path = model_path or os.path.join(config.MODEL_SAVE_DIR, 'best_model.h5')
        if verbose:
            print(f'Loading model from {model_path}')
        model = model_module.load_trained_model(model_path)

    feature_extractor = build_feature_extractor(model)

    if species_data is None:
        species_data = data_loader.load_species_data()
    if background_data is None:
        background_data = data_loader.load_background_data()

    cache_path = (output_dir / 'class_stats.npz') if use_cached_stats else None
    class_means, inv_cov, train_feats_by_class = compute_class_statistics(
        feature_extractor, species_data, background_data,
        cache_path=cache_path, verbose=verbose)
    class_thresholds = calibrate_thresholds(
        train_feats_by_class, class_means, inv_cov, percentile=percentile)

    det_df = load_detection_csvs(detection_dir)
    if verbose:
        print(f'{len(det_df)} detections across {det_df["source_file"].nunique()} files')
    if clips_dir is None:
        # Detections are usually reviewed from exported clips, and those clips
        # hold the same audio the source recordings would be cut down to. When
        # they are present, prefer them: the long recordings are often left
        # behind on the machine that ran the detection, and re-cutting from
        # sources that are not there is the one way this step fails outright.
        default_clips = Path(config.OUTPUT_ROOT) / 'detected_clips'
        if default_clips.is_dir() and any(default_clips.rglob('*.wav')):
            clips_dir = str(default_clips)
            if verbose:
                print(f'  Using the exported clips under {default_clips} '
                      f'(pass clips_dir=... to override, or clips_dir="" to '
                      f'cut from the source recordings instead).')

    if clips_dir:
        # Reuse the clips already exported for manual review; the source
        # recordings are then not needed at all.
        clips = load_clips_from_dir(det_df, clips_dir, padding=clips_padding,
                                    verbose=verbose)
    else:
        clips = extract_clips(det_df, verbose=verbose)

    # One forward pass supplies the features the Mahalanobis and recurrence
    # signals need and the probabilities the margin needs.
    combo = build_feature_extractor(model, with_probs=True)
    X = clips_to_model_input(clips)
    feats, probs = combo.predict(X, batch_size=config.BATCH_SIZE,
                                 verbose=1 if verbose else 0)
    det_df = annotate_softmax_margin(det_df, probs)
    det_df = annotate_station_recurrence(det_df, feats, verbose=verbose)

    det_df = filter_mahalanobis(det_df, clips, feature_extractor, class_means,
                                inv_cov, class_thresholds, percentile=percentile,
                                calibrate_on=mahal_calibration, feats=feats,
                                verbose=verbose)
    if use_yamnet is None:
        use_yamnet = getattr(config, 'USE_YAMNET_FILTER', True)
    if use_yamnet:
        det_df = filter_yamnet(det_df, clips, suspicious=suspicious_yamnet,
                               verbose=verbose)
    else:
        if verbose:
            print('  [off] YAMNet filter disabled (config.USE_YAMNET_FILTER); '
                  'see the note there for the measured reason.')
        det_df = det_df.copy()
        det_df['yamnet_top'] = '(yamnet disabled)'
        det_df['yamnet_score'] = 0.0
        det_df['flag_yamnet'] = False
    det_df = filter_temporal_isolation(det_df, window_s=isolation_window_s, verbose=verbose)
    if getattr(config, 'USE_STATION_REGIME_FILTER', True):
        det_df = filter_station_regime(det_df, verbose=verbose)
    else:
        det_df['station_regime'] = station_regime.NORMAL
        det_df['flag_invading_cluster'] = False
    det_df = merge_flags(det_df)

    clean_df = det_df[det_df['n_flags'] == 0].copy()
    suspicious_df = det_df[det_df['n_flags'] > 0].copy()
    # Corroboration between filters is the usual bar for mining a hard
    # negative, but an invading cluster cannot meet it: the other filters are
    # blind to it by construction, since its detections are neither isolated in
    # time nor outliers in feature space. It is also the clearest hard negative
    # available -- an entire species the model was never trained on -- so it
    # qualifies on its own.
    strong_fp_df = det_df[(det_df['n_flags'] >= 2)
                          | det_df.get('flag_invading_cluster', False)].copy()

    drop = ['all_probs']
    clean_df.drop(columns=drop, errors='ignore').to_csv(
        output_dir / 'clean_detections.csv', index=False)
    suspicious_df.drop(columns=drop, errors='ignore').to_csv(
        output_dir / 'suspicious_detections.csv', index=False)

    # The review queue, over *every* detection rather than only the clean ones.
    # The field evaluation found the ordering and the episode grouping more
    # useful than the clean/suspicious split, and neither of them discards
    # anything, so the queue is built before that split is applied.
    review_queue.save(det_df.drop(columns=drop, errors='ignore'), output_dir,
                      verbose=verbose)

    n_saved = 0
    if save_clips and len(strong_fp_df) > 0:
        n_saved = save_hard_negatives(strong_fp_df, clips, fp_dir)

    n_clean_saved = n_susp_saved = 0
    if save_all_clips:
        if len(clean_df) > 0:
            n_clean_saved = save_clips_by_species(
                clean_df, clips, output_dir / 'clean_clips')
        if len(suspicious_df) > 0:
            n_susp_saved = save_clips_by_species(
                suspicious_df, clips, output_dir / 'suspicious_clips')

    summary = summarize(det_df)
    if verbose:
        print(f'\nClean:      {len(clean_df)}'
              + (f' (saved {n_clean_saved} clips)' if save_all_clips else ''))
        print(f'Suspicious: {len(suspicious_df)}'
              + (f' (saved {n_susp_saved} clips)' if save_all_clips else ''))
        print(f'Strong FPs: {len(strong_fp_df)} (saved {n_saved} clips)')
        print(f'\n{summary.to_string()}')
        print(f'\nResults written to {output_dir}')

    return {
        'det_df': det_df,
        'clean_df': clean_df,
        'suspicious_df': suspicious_df,
        'strong_fp_df': strong_fp_df,
        'summary': summary,
        'class_thresholds': class_thresholds,
        'output_dir': str(output_dir),
    }


if __name__ == '__main__':
    print('Auto-cleanup module. Call run_auto_cleanup() or use '
          'scripts/run_auto_cleanup.py.')
