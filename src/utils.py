"""
Utilities Module
Visualization and helper functions
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pandas as pd
import os
from typing import List, Tuple

try:
    from . import config
except ImportError:  # Allow running as a standalone script (e.g. in Colab)
    import config


def visualize_detection_results(audio_path: str,
                                detections_df: pd.DataFrame,
                                save_path: str = None,
                                show_spectrogram: bool = True):
    """
    Visualize detection results on audio waveform and/or spectrogram
    
    Args:
        audio_path: Path to audio file
        detections_df: DataFrame with detection results
        save_path: Path to save figure (if None, just display)
        show_spectrogram: Whether to show spectrogram or just waveform
    """
    # Load audio
    audio, sr = librosa.load(audio_path, sr=config.SAMPLE_RATE)
    duration = len(audio) / sr
    
    # Create figure
    if show_spectrogram:
        fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(16, 4))
        axes = [axes]
    
    # Plot waveform
    times = np.arange(len(audio)) / sr
    axes[0].plot(times, audio, linewidth=0.5, alpha=0.7, color='blue')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f'Audio Waveform: {os.path.basename(audio_path)}')
    axes[0].set_xlim(0, duration)
    axes[0].grid(True, alpha=0.3)
    
    # Add detection boxes to waveform
    if len(detections_df) > 0:
        # Color map for species (kept in sync with config.SPECIES_FOLDERS)
        species_colors = {
            'Cercopithecus_nictitans': 'orange',
            'Colobus_guereza': 'green',
            'Pan_troglodytes': 'blue',
            'Background': 'gray',
        }

        for _, det in detections_df.iterrows():
            color = species_colors.get(det['species'], 'purple')
            axes[0].axvspan(det['start_time'], det['end_time'], 
                          alpha=0.3, color=color, 
                          label=f"{det['species']} ({det['confidence']:.2f})")
            axes[0].axvline(det['start_time'], color=color, linestyle='--', alpha=0.5)
    
    # Plot spectrogram if requested
    if show_spectrogram:
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
            fmin=config.FMIN,
            fmax=config.FMAX
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        img = librosa.display.specshow(
            mel_spec_db,
            x_axis='time',
            y_axis='mel',
            sr=sr,
            hop_length=config.HOP_LENGTH,
            fmin=config.FMIN,
            fmax=config.FMAX,
            ax=axes[1],
            cmap='viridis'
        )
        
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Frequency (Hz)')
        axes[1].set_title('Mel-Spectrogram with Detections')
        fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
        
        # Add detection boxes to spectrogram
        if len(detections_df) > 0:
            for _, det in detections_df.iterrows():
                color = species_colors.get(det['species'], 'purple')
                axes[1].axvspan(det['start_time'], det['end_time'],
                              alpha=0.3, color=color)
                axes[1].axvline(det['start_time'], color=color, linestyle='--', alpha=0.8, linewidth=2)
    
    # Add legend (remove duplicates)
    if len(detections_df) > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axes[0].legend(by_label.values(), by_label.keys(), 
                      loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_all_detections(detections_dict: dict, 
                             output_dir: str = config.VISUALIZATION_DIR):
    """
    Create visualizations for all detection results
    
    Args:
        detections_dict: Dictionary mapping filename to detection DataFrame
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n Creating visualizations...")
    
    for filename, detections_df in detections_dict.items():
        print(f"\n   Processing: {filename}")
        
        # Get audio path
        audio_path = os.path.join(config.LONG_AUDIO_ROOT, filename)
        
        # Create save path
        base_name = os.path.splitext(filename)[0]
        save_path = os.path.join(output_dir, f"{base_name}_visualization.png")
        
        # Create visualization
        try:
            visualize_detection_results(audio_path, detections_df, save_path)
        except Exception as e:
            print(f"   ⚠️  Error: {e}")
    
    print("\n All visualizations created!")


def plot_confusion_matrix(cm: np.ndarray, 
                         class_names: List[str],
                         save_path: str = None,
                         normalize: bool = False):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save figure
        normalize: Whether to normalize the confusion matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='True label',
           xlabel='Predicted label',
           title='Confusion Matrix')
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                text = f"{cm[i, j]:.1%}"
            else:
                text = f"{cm[i, j]:d}"
            ax.text(j, i, text,
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Confusion matrix saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_detection_summary_report(detections_dict: dict, 
                                   output_path: str = None) -> pd.DataFrame:
    """
    Create a summary report of all detections
    
    Args:
        detections_dict: Dictionary mapping filename to detection DataFrame
        output_path: Path to save CSV report
    
    Returns:
        Summary DataFrame
    """
    summary_data = []
    
    for filename, detections_df in detections_dict.items():
        if len(detections_df) == 0:
            summary_data.append({
                'filename': filename,
                'total_detections': 0,
                'species': 'None',
                'count': 0,
                'avg_confidence': 0.0,
                'min_confidence': 0.0,
                'max_confidence': 0.0
            })
        else:
            # Overall stats
            total = len(detections_df)
            
            # Per-species stats
            for species in detections_df['species'].unique():
                species_df = detections_df[detections_df['species'] == species]
                summary_data.append({
                    'filename': filename,
                    'total_detections': total,
                    'species': species,
                    'count': len(species_df),
                    'avg_confidence': species_df['confidence'].mean(),
                    'min_confidence': species_df['confidence'].min(),
                    'max_confidence': species_df['confidence'].max()
                })
    
    summary_df = pd.DataFrame(summary_data)
    
    if output_path:
        summary_df.to_csv(output_path, index=False)
        print(f"\n Summary report saved to: {output_path}")
    
    return summary_df


def extract_detected_audio_clips(audio_path: str,
                                detections_df: pd.DataFrame,
                                output_dir: str,
                                padding: float = 0.5):
    """
    Extract audio clips for each detection in a single long audio file.

    Args:
        audio_path: Path to original audio file
        detections_df: DataFrame with detections
        output_dir: Directory to save extracted clips
        padding: Padding to add around detection (seconds)
    """
    import soundfile as sf

    os.makedirs(output_dir, exist_ok=True)

    # Load audio
    audio, sr = librosa.load(audio_path, sr=config.SAMPLE_RATE)

    base_name = os.path.splitext(os.path.basename(audio_path))[0]

    print(f"\n Extracting {len(detections_df)} detected clips...")

    for _, det in detections_df.iterrows():
        # Calculate clip boundaries with padding
        start_sample = int(max(0, det['start_time'] - padding) * sr)
        end_sample = int(min(len(audio), det['end_time'] + padding) * sr)

        # Extract clip
        clip = audio[start_sample:end_sample]

        # Create filename
        species = det['species']
        confidence = float(det['confidence'])
        clip_filename = f"{base_name}_{species}_{det['start_time']:.1f}s_conf{confidence:.3f}.wav"
        clip_path = os.path.join(output_dir, clip_filename)

        # Save
        sf.write(clip_path, clip, sr)

    print(f"Saved {len(detections_df)} clips to: {output_dir}")


def extract_all_detected_clips(all_detections: dict,
                               output_dir: str = None,
                               padding: float = 0.5,
                               organize_by_species: bool = True,
                               long_audio_root: str = None) -> str:
    """
    Extract WAV clips for every detection across every long-audio file so they
    can be manually reviewed by listening.

    The clip filenames encode the predicted species, source recording, start
    time (seconds into the source), and the model confidence so a human
    reviewer can sort / filter without consulting any side metadata.

    Args:
        all_detections: Dict mapping long-audio filename -> detections DataFrame
            (the structure returned by ``detection.process_all_long_audio_files``).
        output_dir: Where to save clips. Defaults to ``OUTPUT_ROOT/detected_clips``.
        padding: Extra seconds of audio kept on each side of the detection
            window so the reviewer hears a little context.
        organize_by_species: If True (default), clips are grouped into
            ``output_dir/<species>/`` subfolders so a reviewer can listen to
            all detections of one species back-to-back.
        long_audio_root: Folder to look up source recordings in. Defaults to
            ``config.LONG_AUDIO_ROOT``.

    Returns:
        The output directory path.
    """
    import soundfile as sf

    if output_dir is None:
        output_dir = os.path.join(config.OUTPUT_ROOT, 'detected_clips')
    if long_audio_root is None:
        long_audio_root = config.LONG_AUDIO_ROOT

    os.makedirs(output_dir, exist_ok=True)

    total_clips = 0
    per_species_count = {}
    skipped_files = []

    print("Extracting detection clips for manual review")
    print(f"  output dir: {output_dir}")
    print(f"  padding:    {padding}s on each side")
    print(f"  layout:     {'one folder per species' if organize_by_species else 'flat'}\n")

    for filename, detections_df in all_detections.items():
        if detections_df is None or len(detections_df) == 0:
            continue

        audio_path = os.path.join(long_audio_root, filename)
        if not os.path.exists(audio_path):
            print(f"  Skip {filename}: source recording not found at {audio_path}")
            skipped_files.append(filename)
            continue

        try:
            audio, sr = librosa.load(audio_path, sr=config.SAMPLE_RATE)
        except (FileNotFoundError, PermissionError) as exc:
            print(f"  Skip {filename}: {exc}")
            skipped_files.append(filename)
            continue
        except Exception as exc:  # noqa: BLE001 - librosa backends raise many types
            print(f"  Skip {filename}: decode error ({exc})")
            skipped_files.append(filename)
            continue

        base_name = os.path.splitext(filename)[0]
        n_in_file = 0

        for _, det in detections_df.iterrows():
            species = str(det['species'])
            conf = float(det['confidence'])
            start_t = float(det['start_time'])
            end_t = float(det['end_time'])

            start_sample = int(max(0, start_t - padding) * sr)
            end_sample = int(min(len(audio), end_t + padding) * sr)
            clip = audio[start_sample:end_sample]

            if organize_by_species:
                species_dir = os.path.join(output_dir, species)
                os.makedirs(species_dir, exist_ok=True)
                save_dir = species_dir
            else:
                save_dir = output_dir

            # Filename layout: species first so it sorts naturally; integer
            # second offset so files sort chronologically inside a species
            # folder; confidence last so the reviewer can eyeball quality.
            clip_name = (
                f"{species}__{base_name}__{int(start_t):05d}s"
                f"__conf{conf:.3f}.wav"
            )
            sf.write(os.path.join(save_dir, clip_name), clip, sr)

            n_in_file += 1
            total_clips += 1
            per_species_count[species] = per_species_count.get(species, 0) + 1

        print(f"  {filename}: extracted {n_in_file} clips")

    print(f"\nDone. {total_clips} clips total:")
    for species, count in sorted(per_species_count.items(), key=lambda x: -x[1]):
        print(f"  {species:30s}: {count}")
    if skipped_files:
        print(f"\nSkipped {len(skipped_files)} source files (see warnings above).")

    print(f"\nClips saved under: {output_dir}")
    print("Open this folder in your Drive and listen to verify each detection.")
    return output_dir


def print_detection_statistics(detections_dict: dict):
    """
    Print detailed statistics about detections

    Args:
        detections_dict: Dictionary mapping filename to detection DataFrame
    """
    print("DETECTION STATISTICS")

    # Filter out empty dataframes before concatenating
    non_empty_dfs = [df for df in detections_dict.values() if len(df) > 0]

    if len(non_empty_dfs) == 0:
        print("\nNo detections found.")
        return

    all_detections = pd.concat(non_empty_dfs, ignore_index=True)
    
    print(f"\n Overall Statistics:")
    print(f"   Total Files: {len(detections_dict)}")
    print(f"   Total Detections: {len(all_detections)}")
    print(f"   Average Confidence: {all_detections['confidence'].mean():.4f}")
    print(f"   Confidence Range: [{all_detections['confidence'].min():.4f}, {all_detections['confidence'].max():.4f}]")
    
    print(f"\n Per-Species Statistics:")
    for species in sorted(all_detections['species'].unique()):
        species_df = all_detections[all_detections['species'] == species]
        print(f"   {species:30s}:")
        print(f"      Count: {len(species_df)}")
        print(f"      Avg Confidence: {species_df['confidence'].mean():.4f}")
        print(f"      Confidence Range: [{species_df['confidence'].min():.4f}, {species_df['confidence'].max():.4f}]")
    


if __name__ == "__main__":
    print("Utilities Module Ready!")
    print("Use visualization and analysis functions as needed.")
