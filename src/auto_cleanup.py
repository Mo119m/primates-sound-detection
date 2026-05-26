"""
Automatic false-positive cleanup for detection results.

Runs three independent filters over saved detection CSVs and splits the
detections into "clean" (trustworthy without listening) and "suspicious":

1. Mahalanobis OOD   - feature distance to the predicted class's training
   cluster. A real call lives close to its training cluster; a bird call does
   not.
2. YAMNet cross-check - Google's 521-class audio tagger flags windows whose top
   class is Bird/Insect/Wind/Rain/Speech/etc.
3. Temporal isolation - primates call in bouts, so a detection with no
   same-species neighbour within +/- isolation_window_s is suspicious.

A detection is clean only if all three filters agree. Detections flagged by
>= 2 filters are saved as hard negatives for the next retraining iteration.

Use :func:`run_auto_cleanup` as the single entry point, or call the individual
``filter_*`` helpers for finer control.
"""

import os
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
except ImportError:  # Allow running as a standalone script (e.g. in Colab)
    import config
    import data_loader
    import preprocessing
    import model as model_module


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

def build_feature_extractor(model, layer_name: str = 'dense_256'):
    """Tap an intermediate dense layer to get a per-window feature vector."""
    import tensorflow as tf
    feat_layer = model.get_layer(layer_name)
    return tf.keras.Model(inputs=model.inputs, outputs=feat_layer.output)


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


# DETECTION LOADING + CLIP EXTRACTION

def load_detection_csvs(detection_dir=None) -> pd.DataFrame:
    """
    Load every ``*_detections.csv`` under ``detection_dir`` (recursively, so
    per-station subfolders are included) and attach the resolved source-audio
    path for each detection.
    """
    detection_dir = Path(detection_dir or config.DETECTION_OUTPUT_DIR)
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
        rows.append(df)
    if not rows:
        raise ValueError('All detection CSVs were empty - nothing to clean up.')

    det_df = pd.concat(rows, ignore_index=True)
    det_df['det_id'] = np.arange(len(det_df))
    return det_df


def extract_clips(det_df: pd.DataFrame):
    """Cut one ``WINDOW_SIZE`` clip per detection, caching each source file."""
    clip_len = int(round(config.WINDOW_SIZE * config.SAMPLE_RATE))
    cache = {}
    clips = []
    for row in det_df.itertuples():
        path = row.source_path
        if not path or not os.path.exists(path):
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
    return clips


# THE THREE FILTERS

def filter_mahalanobis(det_df, clips, feature_extractor, class_means, inv_cov,
                       class_thresholds, percentile: int = 95,
                       calibrate_on: str = 'detections',
                       verbose: bool = True) -> pd.DataFrame:
    """Flag detections whose feature vector is OOD.

    calibrate_on='detections' (default) computes per-species thresholds from
    the detection distances themselves, avoiding domain-shift false alarms
    when noisy field recordings differ from clean training clips.
    calibrate_on='training' uses the training-data thresholds (class_thresholds).
    """
    X = np.stack([_audio_to_model_input(c, config.SAMPLE_RATE) for c in clips]).astype(np.float32)
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
    det_df['flag_mahal'] = flags
    if verbose:
        print(f'Mahalanobis flagged {int(flags.sum())} / {len(det_df)} detections')
    return det_df


def filter_yamnet(det_df, clips, suspicious=None, verbose: bool = True) -> pd.DataFrame:
    """Flag detections whose top YAMNet class is a known non-primate sound."""
    import tensorflow as tf
    import tensorflow_hub as hub

    suspicious = suspicious or DEFAULT_SUSPICIOUS_YAMNET
    yamnet = hub.load('https://tfhub.dev/google/yamnet/1')
    class_map_path = yamnet.class_map_path().numpy().decode('utf-8')
    yam_classes = pd.read_csv(class_map_path)['display_name'].tolist()

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
                              verbose: bool = True) -> pd.DataFrame:
    """Flag detections with no same-species neighbour within +/- window_s."""
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
            iso[pos[ids[k]]] = (n == 0)
    det_df['n_neighbours'] = n_neigh
    det_df['flag_isolated'] = iso
    if verbose:
        print(f'Temporal-isolation flagged {int(iso.sum())} / {len(det_df)} detections')
    return det_df


# MERGE + SAVE

def merge_flags(det_df: pd.DataFrame) -> pd.DataFrame:
    """Add ``n_flags`` and a human-readable ``flag_reason`` column."""
    det_df = det_df.copy()
    flag_cols = ['flag_mahal', 'flag_yamnet', 'flag_isolated']
    det_df['n_flags'] = det_df[flag_cols].sum(axis=1).astype(int)

    def reason(row):
        parts = []
        if row.flag_mahal:
            parts.append('mahal')
        if row.flag_yamnet:
            parts.append(f'yamnet:{row.yamnet_top}')
        if row.flag_isolated:
            parts.append('isolated')
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
                     output_dir=None, fp_dir=None,
                     species_data=None, background_data=None,
                     percentile: int = 95, isolation_window_s: float = 30.0,
                     suspicious_yamnet=None, save_clips: bool = True,
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
    clips = extract_clips(det_df)

    det_df = filter_mahalanobis(det_df, clips, feature_extractor, class_means,
                                inv_cov, class_thresholds, percentile=percentile,
                                calibrate_on=mahal_calibration, verbose=verbose)
    det_df = filter_yamnet(det_df, clips, suspicious=suspicious_yamnet, verbose=verbose)
    det_df = filter_temporal_isolation(det_df, window_s=isolation_window_s, verbose=verbose)
    det_df = merge_flags(det_df)

    clean_df = det_df[det_df['n_flags'] == 0].copy()
    suspicious_df = det_df[det_df['n_flags'] > 0].copy()
    strong_fp_df = det_df[det_df['n_flags'] >= 2].copy()

    drop = ['all_probs']
    clean_df.drop(columns=drop, errors='ignore').to_csv(
        output_dir / 'clean_detections.csv', index=False)
    suspicious_df.drop(columns=drop, errors='ignore').to_csv(
        output_dir / 'suspicious_detections.csv', index=False)

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
