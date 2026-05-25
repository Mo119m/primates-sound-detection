"""
Detection Module
Detect primate vocalizations in long audio files using sliding window approach
"""

import numpy as np
import os
from typing import List, Tuple, Dict
import pandas as pd

try:
    from . import config
    from . import preprocessing
    from . import data_loader
except ImportError:  # Allow running as a standalone script (e.g. in Colab)
    import config
    import preprocessing
    import data_loader


def get_detection_groups() -> Tuple[List[str], Dict[str, List[int]]]:
    """
    Build the ordered detection labels (Background last) and, for each, the list
    of CLASS_NAMES indices whose softmax scores feed it.

    Returns:
        Tuple of (labels, indices) where labels is the ordered group names and
        indices maps each group to the CLASS_NAMES positions it aggregates.
    """
    labels: List[str] = []
    for name in config.CLASS_NAMES:
        group = config.DETECTION_GROUPS.get(name, name)
        if group not in labels:
            labels.append(group)
    if 'Background' in labels:
        labels.remove('Background')
        labels.append('Background')

    indices = {
        group: [i for i, name in enumerate(config.CLASS_NAMES)
                if config.DETECTION_GROUPS.get(name, name) == group]
        for group in labels
    }
    return labels, indices


def group_probabilities(pred: np.ndarray,
                        labels: List[str],
                        indices: Dict[str, List[int]]) -> np.ndarray:
    """
    Collapse a CLASS_NAMES-ordered softmax vector onto detection groups by
    summing the probabilities of member classes.

    Args:
        pred: softmax scores for one window, in CLASS_NAMES order
        labels: ordered group names from get_detection_groups
        indices: group -> member class indices from get_detection_groups

    Returns:
        Array of grouped probabilities aligned with labels.
    """
    pred = np.asarray(pred)
    return np.array([pred[indices[g]].sum() for g in labels])


def detect_in_long_audio(model,
                         audio_path: str,
                         confidence_threshold: float = config.DETECTION_CONFIDENCE_THRESHOLD) -> pd.DataFrame:
    """
    Detect primate vocalizations in a long audio file
    
    Args:
        model: Trained Keras model
        audio_path: Path to long audio file
        confidence_threshold: Minimum confidence for detection
    
    Returns:
        DataFrame with detection results
    """
  
    print(f"DETECTING IN: {os.path.basename(audio_path)}")
    
    # Load long audio
    audio, sr = data_loader.load_long_audio(audio_path)
    if audio is None:
        return pd.DataFrame()
    
    # Extract sliding windows
    print(f"\n Extracting sliding windows")
    print(f"   Window size: {config.WINDOW_SIZE}s")
    print(f"   Stride: {config.WINDOW_STRIDE}s")
    
    windows, times = preprocessing.extract_sliding_windows(audio, sr)
    print(f"   Total windows: {len(windows)}")
    
    # Preprocess windows
    print(f"\n Preprocessing windows")
    X_windows = []
    for i, window in enumerate(windows):
        img = preprocessing.preprocess_audio(window, sr)
        img_norm = preprocessing.preprocess_for_model(img)
        X_windows.append(img_norm)
        
        if (i + 1) % 100 == 0:
            print(f"   Processed {i + 1}/{len(windows)}...")
    
    X_windows = np.array(X_windows)
    print(f" Preprocessed {len(X_windows)} windows")
    
    # Run predictions
    print(f"\n Running predictions")
    predictions = model.predict(X_windows, batch_size=config.BATCH_SIZE, verbose=1)
    
    # Process predictions
    print(f"\n Processing detections")
    detections = []

    # Collapse the fine-grained classes onto detection groups so the three
    # Cernic call types vote together for a single "Cernic" detection.
    labels, group_indices = get_detection_groups()

    for i, (pred, (start_time, end_time)) in enumerate(zip(predictions, times)):
        grouped = group_probabilities(pred, labels, group_indices)
        top = int(np.argmax(grouped))
        label = labels[top]
        confidence = float(grouped[top])

        # Only keep non-background detections above threshold
        if label != 'Background' and confidence >= confidence_threshold:
            detections.append({
                'start_time': start_time,
                'end_time': end_time,
                'species': label,
                'confidence': confidence,
                'all_probs': pred.tolist()
            })
    
    print(f"   Found {len(detections)} detections above threshold")
    
    # Apply Non-Maximum Suppression
    if len(detections) > 0:
        print(f"\n Applying Non-Maximum Suppression")
        detections = apply_nms(detections)
        print(f"   After NMS: {len(detections)} detections")
    
    # Convert to DataFrame
    df = pd.DataFrame(detections)
    
    # Print summary
    if len(df) > 0:
        print("\n Detection Summary:")

        for species in df['species'].unique():
            count = len(df[df['species'] == species])
            avg_conf = df[df['species'] == species]['confidence'].mean()
            print(f"   {species:30s}: {count:3d} detections (avg conf: {avg_conf:.4f})")

    else:
        print("\n   No detections found.")
    

    return df


def run_raw_inference(model, audio_path: str) -> Tuple[List[Tuple[float, float]], np.ndarray]:
    """
    Run sliding-window inference on a long audio file without applying any
    threshold or NMS. Useful when you want to compare several thresholds
    without re-running the (expensive) forward pass.

    Args:
        model: Trained Keras model
        audio_path: Path to long audio file

    Returns:
        Tuple of (times, probs)
        - times: list of (start_time, end_time) per window
        - probs: np.ndarray of shape (n_windows, n_classes) with softmax scores
    """
    print(f"RAW INFERENCE: {os.path.basename(audio_path)}")

    audio, sr = data_loader.load_long_audio(audio_path)
    if audio is None:
        return [], np.zeros((0, len(config.CLASS_NAMES)), dtype=np.float32)

    windows, times = preprocessing.extract_sliding_windows(audio, sr)
    print(f"   windows: {len(windows)}")

    X_windows = []
    for window in windows:
        img = preprocessing.preprocess_audio(window, sr)
        img_norm = preprocessing.preprocess_for_model(img)
        X_windows.append(img_norm)
    X_windows = np.array(X_windows)

    probs = model.predict(X_windows, batch_size=config.BATCH_SIZE, verbose=1)
    return times, probs


def predictions_to_detections(times: List[Tuple[float, float]],
                              probs: np.ndarray,
                              confidence_threshold: float,
                              apply_nms_filter: bool = True) -> pd.DataFrame:
    """
    Convert raw sliding-window predictions into a detections DataFrame at a
    given confidence threshold. Background class is always excluded.

    Args:
        times: list of (start_time, end_time) from run_raw_inference
        probs: (n_windows, n_classes) softmax scores from run_raw_inference
        confidence_threshold: minimum confidence to keep a window
        apply_nms_filter: whether to run per-species NMS after thresholding

    Returns:
        DataFrame with columns start_time, end_time, species, confidence,
        all_probs.
    """
    if len(times) == 0 or len(probs) == 0:
        return pd.DataFrame(columns=['start_time', 'end_time', 'species',
                                     'confidence', 'all_probs'])

    labels, group_indices = get_detection_groups()
    detections = []
    for pred, (start_time, end_time) in zip(probs, times):
        grouped = group_probabilities(pred, labels, group_indices)
        top = int(np.argmax(grouped))
        label = labels[top]
        confidence = float(grouped[top])
        if label == 'Background':
            continue
        if confidence < confidence_threshold:
            continue
        detections.append({
            'start_time': float(start_time),
            'end_time': float(end_time),
            'species': label,
            'confidence': confidence,
            'all_probs': pred.tolist(),
        })

    if apply_nms_filter and len(detections) > 0:
        detections = apply_nms(detections)

    return pd.DataFrame(detections)


def run_raw_inference_all(model) -> Dict[str, Tuple[List[Tuple[float, float]], np.ndarray]]:
    """
    Run raw sliding-window inference on every long-audio file exactly once.
    The returned dict can then be fed through predictions_to_detections at
    several thresholds (threshold sweep) without re-running the model.

    Returns:
        Dict mapping filename -> (times, probs)
    """
    print("RAW INFERENCE ON ALL LONG AUDIO FILES")
    audio_files = data_loader.get_long_audio_files()
    print(f"\nFound {len(audio_files)} audio files to process")

    raw = {}
    for i, audio_path in enumerate(audio_files, 1):
        print(f"\nFile {i}/{len(audio_files)}")
        times, probs = run_raw_inference(model, audio_path)
        raw[os.path.basename(audio_path)] = (times, probs)

    total_windows = sum(len(t) for t, _ in raw.values())
    print(f"\nDone. {total_windows} windows across {len(raw)} files.")
    return raw


def sweep_thresholds(raw_results: Dict[str, Tuple[List[Tuple[float, float]], np.ndarray]],
                     thresholds: List[float],
                     apply_nms_filter: bool = True) -> Dict[float, Dict[str, pd.DataFrame]]:
    """
    Apply several confidence thresholds to a set of pre-computed raw
    predictions and return the per-threshold detection DataFrames.

    Args:
        raw_results: output of run_raw_inference_all
        thresholds: list of confidence cutoffs to evaluate
        apply_nms_filter: whether to run NMS at each threshold

    Returns:
        Dict mapping threshold -> {filename: detections DataFrame}
    """
    sweep: Dict[float, Dict[str, pd.DataFrame]] = {}
    for thr in thresholds:
        per_file = {}
        for fname, (times, probs) in raw_results.items():
            per_file[fname] = predictions_to_detections(
                times, probs, confidence_threshold=thr,
                apply_nms_filter=apply_nms_filter,
            )
        sweep[thr] = per_file
    return sweep


def apply_nms(detections: List[Dict],
              iou_threshold: float = config.NMS_IOU_THRESHOLD) -> List[Dict]:
    """
    Apply Non-Maximum Suppression to remove overlapping detections
    
    Args:
        detections: List of detection dictionaries
        iou_threshold: IoU threshold for suppression
    
    Returns:
        Filtered list of detections
    """
    if len(detections) == 0:
        return []
    
    # Group detections by species
    species_groups = {}
    for det in detections:
        species = det['species']
        if species not in species_groups:
            species_groups[species] = []
        species_groups[species].append(det)
    
    # Apply NMS per species
    all_kept = []
    for species, dets in species_groups.items():
        # Sort by confidence (descending)
        dets = sorted(dets, key=lambda x: x['confidence'], reverse=True)
        
        kept = []
        while len(dets) > 0:
            # Keep the highest confidence detection
            best = dets.pop(0)
            kept.append(best)
            
            # Remove overlapping detections
            dets = [d for d in dets if compute_iou(best, d) < iou_threshold]
        
        all_kept.extend(kept)
    
    # Sort by start time
    all_kept = sorted(all_kept, key=lambda x: x['start_time'])
    
    return all_kept


def compute_iou(det1: Dict, det2: Dict) -> float:
    """
    Compute Intersection over Union (IoU) between two detections
    
    Args:
        det1: First detection
        det2: Second detection
    
    Returns:
        IoU value
    """
    # Get time ranges
    start1, end1 = det1['start_time'], det1['end_time']
    start2, end2 = det2['start_time'], det2['end_time']
    
    # Compute intersection
    inter_start = max(start1, start2)
    inter_end = min(end1, end2)
    intersection = max(0, inter_end - inter_start)
    
    # Compute union
    union = (end1 - start1) + (end2 - start2) - intersection
    
    # Compute IoU
    iou = intersection / union if union > 0 else 0
    
    return iou


def save_detections(detections_df: pd.DataFrame, 
                   audio_filename: str,
                   output_dir: str = config.DETECTION_OUTPUT_DIR) -> str:
    """
    Save detection results to CSV file
    
    Args:
        detections_df: DataFrame with detections
        audio_filename: Name of the audio file
        output_dir: Directory to save results
    
    Returns:
        Path to saved CSV file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output filename
    base_name = os.path.splitext(audio_filename)[0]
    csv_path = os.path.join(output_dir, f"{base_name}_detections.csv")
    
    # Save to CSV
    if len(detections_df) > 0:
        # Select columns to save
        save_df = detections_df[['start_time', 'end_time', 'species', 'confidence']].copy()
        save_df.to_csv(csv_path, index=False)
        print(f"\n Detections saved to: {csv_path}")
    else:
        # Save empty file with headers
        pd.DataFrame(columns=['start_time', 'end_time', 'species', 'confidence']).to_csv(csv_path, index=False)
        print(f"\n No detections, empty file saved to: {csv_path}")
    
    return csv_path


def process_all_long_audio_files(model, 
                                 confidence_threshold: float = config.DETECTION_CONFIDENCE_THRESHOLD) -> Dict[str, pd.DataFrame]:
    """
    Process all long audio files in the directory
    
    Args:
        model: Trained model
        confidence_threshold: Confidence threshold for detections
    
    Returns:
        Dictionary mapping filename to detection DataFrame
    """
    print("PROCESSING ALL LONG AUDIO FILES")
    
    # Get all long audio files
    audio_files = data_loader.get_long_audio_files()
    print(f"\nFound {len(audio_files)} audio files to process")
    
    all_detections = {}
    
    for i, audio_path in enumerate(audio_files, 1):
        print(f"Processing file {i}/{len(audio_files)}")
        
        # Detect
        detections_df = detect_in_long_audio(model, audio_path, confidence_threshold)
        
        # Save
        filename = os.path.basename(audio_path)
        save_detections(detections_df, filename)
        
        # Store
        all_detections[filename] = detections_df
    

    print(" ALL FILES PROCESSED!")
    
    # Print overall summary
    print("\n Overall Summary:")
    total_detections = 0
    for filename, df in all_detections.items():
        count = len(df)
        total_detections += count
        print(f"   {filename:40s}: {count:4d} detections")
    print(f"   TOTAL: {total_detections} detections across {len(audio_files)} files")
    
    return all_detections


if __name__ == "__main__":
    print("Detection Module Ready!")
    print("Load a trained model and call detect_in_long_audio() or process_all_long_audio_files().")
