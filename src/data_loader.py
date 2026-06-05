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


def find_loudest_window(audio: np.ndarray, target_length: int) -> int:
    """
    Return the start sample of the highest-energy window of target_length.

    The training clips (e.g. 5 s putty-nose) do not call for their whole
    duration, so a random 2 s crop can land on silence and mislabel it as a
    call. Picking the window with maximum short-time energy keeps the crop on
    the actual vocalisation.

    Args:
        audio: 1-D waveform
        target_length: window length in samples

    Returns:
        Start index of the loudest window (0 if audio is not longer than the
        window).
    """
    if len(audio) <= target_length:
        return 0
    power = audio.astype(np.float64) ** 2
    cumulative = np.concatenate(([0.0], np.cumsum(power)))
    # Energy of the window starting at s is cumulative[s+L] - cumulative[s].
    window_energy = cumulative[target_length:] - cumulative[:-target_length]
    return int(np.argmax(window_energy))


def embed_in_background(call: np.ndarray,
                        target_length: int,
                        background_pool: List[np.ndarray],
                        snr_db_range: Tuple[float, float] = (3.0, 15.0)) -> np.ndarray:
    """
    Place a short call at a random position inside a real background bed.

    Zero-padding a short call to the clip length leaves a digital-silence
    region that never occurs in field recordings and gives the model a
    high-contrast "impulse against black" shortcut (shared by gunshots and
    branch-snaps). Embedding the call in genuine forest ambient instead makes
    the training clip look like an actual detection window that contains a call.

    Args:
        call: short call waveform (shorter than target_length)
        target_length: output length in samples
        background_pool: list of real background waveforms to draw the bed from
        snr_db_range: call-to-background SNR drawn uniformly so the call stays
            the dominant sound while ambient fills the rest of the window

    Returns:
        target_length waveform: background bed with the call added on top.
    """
    bg = background_pool[np.random.randint(len(background_pool))].astype(np.float32)

    # Make the bed exactly target_length (tile if short, random-crop if long).
    if len(bg) < target_length:
        reps = int(np.ceil(target_length / max(len(bg), 1)))
        bg = np.tile(bg, reps)[:target_length]
    elif len(bg) > target_length:
        start = np.random.randint(0, len(bg) - target_length + 1)
        bg = bg[start:start + target_length]
    bg = bg.copy()

    # Scale the bed to sit a realistic SNR below the call.
    call_rms = float(np.sqrt(np.mean(call.astype(np.float64) ** 2))) + 1e-8
    bg_rms = float(np.sqrt(np.mean(bg.astype(np.float64) ** 2))) + 1e-8
    snr_db = np.random.uniform(*snr_db_range)
    target_bg_rms = call_rms / (10 ** (snr_db / 20))
    bg *= target_bg_rms / bg_rms

    # Drop the call at a random offset (a call can fall anywhere in a window).
    max_offset = target_length - len(call)
    offset = np.random.randint(0, max_offset + 1) if max_offset > 0 else 0
    bg[offset:offset + len(call)] += call.astype(np.float32)
    return bg


def load_audio_file(file_path: str,
                    target_sr: int = config.SAMPLE_RATE,
                    target_duration: float = config.CLIP_DURATION,
                    crop: str = 'loudest',
                    background_pool: List[np.ndarray] = None):
    """
    Load a single audio file and ensure consistent length.

    Files longer than the target are cropped to target_duration according to
    ``crop``:
      - 'loudest' (default): the highest-energy window, so the crop lands on
        the call even when the source clip has silent stretches.
      - 'random': a random window (light augmentation; may land on silence).
      - 'start': the leading segment.

    Files shorter than the target are extended to the clip length. When
    ``background_pool`` is given the short call is embedded in a real background
    bed (see :func:`embed_in_background`); otherwise it is zero-padded.

    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
        target_duration: Target duration in seconds
        crop: Cropping strategy for files longer than the target
        background_pool: optional real background waveforms; when provided,
            short clips are embedded in ambient instead of zero-padded

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

    # Extend (short) or crop (long) to the target length
    if len(audio) < target_length:
        if background_pool:
            audio = embed_in_background(audio, target_length, background_pool)
        else:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
    elif len(audio) > target_length:
        if crop == 'loudest':
            start = find_loudest_window(audio, target_length)
        elif crop == 'random':
            start = np.random.randint(0, len(audio) - target_length + 1)
        else:  # 'start'
            start = 0
        audio = audio[start:start + target_length]

    return audio


def load_species_data(background_pool: List[np.ndarray] = None) -> Dict[str, List[Tuple[np.ndarray, str]]]:
    """
    Load all species audio data

    Args:
        background_pool: optional real background waveforms. When given, short
            clips (e.g. hack/kek/pyow) are embedded in ambient instead of
            zero-padded, removing the silence shortcut. Load background first
            and pass its waveforms here.

    Returns:
        Dictionary mapping species name to list of (audio, file_path) tuples
    """
    species_data = {}

    print("\n Loading Species Data")

    for species_name, folder_names in config.SPECIES_FOLDERS.items():
        print(f"\n loading {species_name}...")

        if isinstance(folder_names, str):
            folder_names = [folder_names]

        audio_files = []
        for folder_name in folder_names:
            found = scan_audio_files(config.AUDIO_ROOT, folder_name)
            print(f"   {folder_name}: {len(found)} files")
            audio_files.extend(found)
        print(f"   Total: {len(audio_files)} files")

        audio_data = []
        for i, file_path in enumerate(audio_files):
            audio = load_audio_file(file_path, background_pool=background_pool)
            if audio is not None:
                audio_data.append((audio, file_path))

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

    Supports two AudioMoth naming schemes:
      - Old: ``YYYYMMDDTHHMMSS+ZZZZ_....wav``
      - New: ``SYYYYMMDDTHHMMSSmmm+ZZZZ_EYYYYMMDD..._<gps>.wav`` where the
        leading ``S`` block is the start time (millisecond precision).

    Returns:
        (hour, minute) as ints, or None if parsing fails.
    """
    import re
    base = os.path.basename(filename)
    m = re.match(r'S\d{8}T(\d{2})(\d{2})', base)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.match(r'\d{8}T(\d{2})(\d{2})', base)
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
