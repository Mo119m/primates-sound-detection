"""
Data Loader Module
Automatically scans and loads audio files from specified folders.
Handles both species clips and background noise.
"""

import os
import numpy as np
import librosa
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from . import config
except ImportError:  # Allow running as a standalone script (e.g. in Colab)
    import config

def scan_audio_files(root_dir: str, folder_name: str) -> List[str]:
    """
    Scan a folder for audio files (.wav)
    
    Args:
        root_dir: Root directory containing audio folders
        folder_name: Name of the specific folder to scan
    
    Returns:
        List of full paths to audio files
    """
    folder_path = os.path.join(root_dir, folder_name)
    
    if not os.path.exists(folder_path):
        print(f" Warning: Folder not found: {folder_path}")
        return []
    
    # Find all .wav files
    audio_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    
    return sorted(audio_files)


def load_audio_file(file_path: str,
                    target_sr: int = config.SAMPLE_RATE,
                    target_duration: float = config.CLIP_DURATION):
    """
    Load a single audio file and ensure consistent length

    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
        target_duration: Target duration in seconds

    Returns:
        Audio waveform as numpy array, or None if the file could not be loaded.
    """
    try:
        # Load audio. librosa raises (FileNotFoundError, sf.SoundFileError, ...)
        # but the exact set depends on the backend, so we catch the common
        # subclasses explicitly and let truly unexpected errors propagate.
        audio, sr = librosa.load(file_path, sr=target_sr)
    except (FileNotFoundError, PermissionError) as exc:
        print(f" Error loading {file_path}: {exc}")
        return None
    except Exception as exc:  # noqa: BLE001 - librosa backends raise many types
        print(f" Error decoding {file_path}: {exc}")
        return None

    # Calculate target length in samples
    target_length = int(target_sr * target_duration)

    # Pad or trim to target length
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
    elif len(audio) > target_length:
        audio = audio[:target_length]

    return audio


def load_species_data() -> Dict[str, List[Tuple[np.ndarray, str]]]:
    """
    Load all species audio data
    
    Returns:
        Dictionary mapping species name to list of (audio, file_path) tuples
    """
    species_data = {}
    
    print("\n Loading Species Data")
    
    for species_name, folder_name in config.SPECIES_FOLDERS.items():
        print(f"\n loading {species_name}...")
        
        # Scan for audio files
        audio_files = scan_audio_files(config.AUDIO_ROOT, folder_name)
        print(f"   Found {len(audio_files)} files")
        
        # Load audio data
        audio_data = []
        for i, file_path in enumerate(audio_files):
            audio = load_audio_file(file_path)
            if audio is not None:
                audio_data.append((audio, file_path))
            
            # Progress indicator
            if (i + 1) % 50 == 0:
                print(f"   Loaded {i + 1}/{len(audio_files)}...")
        
        species_data[species_name] = audio_data
        print(f" Successfully loaded {len(audio_data)} clips")
    
    return species_data


def load_background_data() -> List[Tuple[np.ndarray, str]]:
    """
    Load all background noise data from multiple folders
    
    Returns:
        List of (audio, file_path) tuples
    """
    background_data = []
    
    print("\n Loading Background Data")
    
    for folder_name in config.BACKGROUND_FOLDERS:
        print(f"\n Loading from {folder_name}...")
        
        # Scan for audio files
        audio_files = scan_audio_files(config.AUDIO_ROOT, folder_name)
        print(f"   Found {len(audio_files)} files")
        
        # Load audio data
        for i, file_path in enumerate(audio_files):
            audio = load_audio_file(file_path)
            if audio is not None:
                background_data.append((audio, file_path))
            
            # Progress indicator
            if (i + 1) % 50 == 0:
                print(f"   Loaded {i + 1}/{len(audio_files)}...")
        
        print(f"Successfully loaded {len(audio_files)} clips")
    
    print(f"\n Total Background Samples: {len(background_data)}")
    
    return background_data


def load_long_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """
    Load a long audio file for detection
    
    Args:
        file_path: Path to long audio file
    
    Returns:
        Tuple of (audio waveform, sample rate)
    """
    print(f"\n Loading long audio: {os.path.basename(file_path)}")
    
    try:
        audio, sr = librosa.load(file_path, sr=config.SAMPLE_RATE)
        duration = len(audio) / sr
        print(f"   Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        print(f"   Sample Rate: {sr} Hz")
        print(f"   Samples: {len(audio):,}")
        return audio, sr
    
    except Exception as e:
        print(f" Error loading long audio: {e}")
        return None, None


def get_long_audio_files(root: str = None) -> List[str]:
    """
    Get list of all long audio files to process.
    Searches recursively so nested layouts (station/date/files) are supported.

    Args:
        root: Directory to scan. Defaults to config.LONG_AUDIO_ROOT.

    Returns:
        List of file paths
    """
    root = root or config.LONG_AUDIO_ROOT
    if not os.path.exists(root):
        print(f" Warning: Long audio folder not found: {root}")
        return []

    audio_files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith('.wav'):
                audio_files.append(os.path.join(dirpath, f))

    return sorted(audio_files)


def parse_recording_time(filename: str):
    """
    Extract the start hour and minute from an IPA recording filename.
    Expected format: YYYYMMDDTHHMMSS+ZZZZ_....wav

    Returns:
        (hour, minute) as ints, or None if parsing fails.
    """
    import re
    base = os.path.basename(filename)
    m = re.match(r'\d{8}T(\d{2})(\d{2})\d{2}', base)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def filter_files_by_time(file_list: List[str],
                         start: str = None,
                         end: str = None) -> List[str]:
    """
    Keep only recordings whose start time falls within [start, end).

    Args:
        file_list: list of WAV file paths
        start: "HH:MM" string (inclusive). Defaults to config.TIME_FILTER_START.
        end:   "HH:MM" string (inclusive). Defaults to config.TIME_FILTER_END.

    Returns:
        Filtered list of file paths.
    """
    start = start or config.TIME_FILTER_START
    end = end or config.TIME_FILTER_END
    if start is None or end is None:
        return file_list

    sh, sm = map(int, start.split(':'))
    eh, em = map(int, end.split(':'))
    start_min = sh * 60 + sm
    end_min = eh * 60 + em

    filtered = []
    for f in file_list:
        t = parse_recording_time(f)
        if t is None:
            continue
        fmin = t[0] * 60 + t[1]
        if start_min <= fmin <= end_min:
            filtered.append(f)
    return sorted(filtered)


def get_ipa_station_files(station: str, time_filter: bool = True) -> List[str]:
    """
    Get all WAV files for a specific IPA station, optionally filtered by time.

    Args:
        station: station folder name, e.g. "IPA1ST"
        time_filter: if True, apply the time window from config

    Returns:
        Sorted list of file paths
    """
    station_dir = os.path.join(config.IPA_ROOT, station)
    files = get_long_audio_files(root=station_dir)
    print(f"  {station}: {len(files)} total WAV files")
    if time_filter:
        files = filter_files_by_time(files)
        print(f"  After time filter ({config.TIME_FILTER_START}–{config.TIME_FILTER_END}): {len(files)} files")
    return files


def print_data_summary(species_data: Dict, background_data: List):
    """
    Print a summary of loaded data
    
    Args:
        species_data: Dictionary of species audio data
        background_data: List of background audio data
    """
    print("DATA LOADING SUMMARY")
    
    total_species_samples = 0
    for species_name, audio_list in species_data.items():
        count = len(audio_list)
        total_species_samples += count
        print(f" {species_name}: {count} samples")
    
    print(f" Background: {len(background_data)} samples")
    print(f"\n Total Samples: {total_species_samples + len(background_data)}")
    
    # Calculate expected samples after augmentation
    print(f"\n After Augmentation (×{config.AUGMENTATION_MULTIPLIER}):")
    for species_name, audio_list in species_data.items():
        original_count = len(audio_list)
        augmented_count = original_count * config.AUGMENTATION_MULTIPLIER
        print(f"   {species_name}: {original_count} -> ~{augmented_count}")
    
    print(f"   Background: {len(background_data)} (no augmentation)")
    


if __name__ == "__main__":
    # Test data loading
    print("Testing Data Loader...")
    config.print_config_summary()
    
    species_data = load_species_data()
    background_data = load_background_data()
    
    print_data_summary(species_data, background_data)
    
    # Test long audio file scanning
    long_audio_files = get_long_audio_files()
    print(f"\n Found {len(long_audio_files)} long audio files:")
    for i, file in enumerate(long_audio_files[:5], 1):  # Show first 5
        print(f"   {i}. {os.path.basename(file)}")
    if len(long_audio_files) > 5:
        print(f"   ... and {len(long_audio_files) - 5} more")
