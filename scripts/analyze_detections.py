"""
Analyze detection results to identify false positives
"""
import os
import sys
sys.path.append('../src')

import pandas as pd
import config

def analyze_species_detections(all_detections, target_species='Cercopithecus_nictitans'):
    """
    Analyze detections for a specific species

    Args:
        all_detections: Dictionary of detection results
        target_species: Species to analyze
    """
    print(f"\n=== Analyzing {target_species} Detections ===\n")

    species_detections = []

    for filename, detections_df in all_detections.items():
        species_df = detections_df[detections_df['species'] == target_species]
        if len(species_df) > 0:
            species_detections.append({
                'filename': filename,
                'count': len(species_df),
                'avg_confidence': species_df['confidence'].mean(),
                'min_confidence': species_df['confidence'].min(),
                'max_confidence': species_df['confidence'].max()
            })

    if len(species_detections) == 0:
        print(f"No {target_species} detections found.")
        return

    analysis_df = pd.DataFrame(species_detections)
    analysis_df = analysis_df.sort_values('count', ascending=False)

    print(f"Total files with {target_species}: {len(analysis_df)}")
    print(f"Total detections: {analysis_df['count'].sum()}")
    print(f"\nTop 10 files with most {target_species} detections:")
    print(analysis_df.head(10).to_string(index=False))

    # Confidence distribution
    print(f"\n\nConfidence Distribution:")
    all_confs = []
    for filename, detections_df in all_detections.items():
        species_df = detections_df[detections_df['species'] == target_species]
        if len(species_df) > 0:
            all_confs.extend(species_df['confidence'].tolist())

    import numpy as np
    print(f"  Mean: {np.mean(all_confs):.4f}")
    print(f"  Median: {np.median(all_confs):.4f}")
    print(f"  Std: {np.std(all_confs):.4f}")
    print(f"  Range: [{np.min(all_confs):.4f}, {np.max(all_confs):.4f}]")

    # Confidence bins
    bins = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    hist, _ = np.histogram(all_confs, bins=bins)
    print(f"\n  Confidence Bins:")
    for i in range(len(bins)-1):
        print(f"    {bins[i]:.2f}-{bins[i+1]:.2f}: {hist[i]} detections ({hist[i]/len(all_confs)*100:.1f}%)")

    return analysis_df


def suggest_threshold(all_detections, target_species, desired_reduction=0.5):
    """
    Suggest a confidence threshold to reduce false positives

    Args:
        all_detections: Dictionary of detection results
        target_species: Species to analyze
        desired_reduction: Target reduction ratio (0.5 = reduce by 50%)
    """
    import numpy as np

    all_confs = []
    for filename, detections_df in all_detections.items():
        species_df = detections_df[detections_df['species'] == target_species]
        if len(species_df) > 0:
            all_confs.extend(species_df['confidence'].tolist())

    if len(all_confs) == 0:
        print(f"No {target_species} detections found.")
        return

    all_confs = np.array(all_confs)
    original_count = len(all_confs)
    target_count = int(original_count * (1 - desired_reduction))

    # Find threshold that keeps target_count highest confidence detections
    threshold = np.sort(all_confs)[::-1][target_count-1] if target_count > 0 else 1.0

    print(f"\n=== Threshold Recommendation ===")
    print(f"Current detections: {original_count}")
    print(f"Target reduction: {desired_reduction*100:.0f}%")
    print(f"Target detections: {target_count}")
    print(f"\nRecommended threshold: {threshold:.4f}")
    print(f"This would keep the top {target_count} highest-confidence detections")

    return threshold


if __name__ == "__main__":
    print("Use this script in your notebook:")
    print("  import analyze_detections")
    print("  analyze_detections.analyze_species_detections(all_detections, 'Cercopithecus_nictitans')")
    print("  analyze_detections.suggest_threshold(all_detections, 'Cercopithecus_nictitans', 0.5)")
