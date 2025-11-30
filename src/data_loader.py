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
                    target_duration: float = config.CLIP_DURATION) -> np.ndarray:
    """
    Load a single audio file and ensure consistent length
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
        target_duration: Target duration in seconds
    
    Returns:
        Audio waveform as numpy array
    """
    try:
        # Load audio
        audio, sr = librosa.load(file_path, sr=target_sr)
        
        # Calculate target length in samples
        target_length = int(target_sr * target_duration)
        
        # Pad or trim to target length
        if len(audio) < target_length:
            # Pad with zeros if too short
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        elif len(audio) > target_length:
            # Trim if too long
            audio = audio[:target_length]
        
        return audio
    
    except Exception as e:
        print(f" Error loading {file_path}: {e}")
        return None


def load_species_data() -> Dict[str, List[Tuple[np.ndarray, str]]]:
    """
    Load all species audio data
    
    Returns:
        Dictionary mapping species name to list of (audio, file_path) tuples
    """
    species_data = {}
    
    print("\n Loading Species Data...")
    print("=" * 70)
    
    for species_name, folder_name in config.SPECIES_FOLDERS.items():
        print(f"\n🐵 Loading {species_name}...")
        
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
    
    print("\n Loading Background Data...")
    print("=" * 70)
    
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


def get_long_audio_files() -> List[str]:
    """
    Get list of all long audio files to process
    
    Returns:
        List of file paths
    """
    if not os.path.exists(config.LONG_AUDIO_ROOT):
        print(f" Warning: Long audio folder not found: {config.LONG_AUDIO_ROOT}")
        return []
    
    audio_files = []
    for file in os.listdir(config.LONG_AUDIO_ROOT):
        if file.lower().endswith('.wav'):
            audio_files.append(os.path.join(config.LONG_AUDIO_ROOT, file))
    
    return sorted(audio_files)


def print_data_summary(species_data: Dict, background_data: List):
    """
    Print a summary of loaded data
    
    Args:
        species_data: Dictionary of species audio data
        background_data: List of background audio data
    """
    print("\n" + "=" * 70)
    print("DATA LOADING SUMMARY")
    print("=" * 70)
    
    total_species_samples = 0
    for species_name, audio_list in species_data.items():
        count = len(audio_list)
        total_species_samples += count
        print(f"🐵 {species_name}: {count} samples")
    
    print(f"🔇 Background: {len(background_data)} samples")
    print(f"\n📊 Total Samples: {total_species_samples + len(background_data)}")
    
    # Calculate expected samples after augmentation
    print(f"\n🔄 After Augmentation (×{config.AUGMENTATION_MULTIPLIER}):")
    for species_name, audio_list in species_data.items():
        original_count = len(audio_list)
        augmented_count = original_count * config.AUGMENTATION_MULTIPLIER
        print(f"   {species_name}: {original_count} → ~{augmented_count}")
    
    print(f"   Background: {len(background_data)} (no augmentation)")
    
    print("=" * 70)


if __name__ == "__main__":
    # Test data loading
    print("Testing Data Loader...")
    config.print_config_summary()
    
    species_data = load_species_data()
    background_data = load_background_data()
    
    print_data_summary(species_data, background_data)
    
    # Test long audio file scanning
    long_audio_files = get_long_audio_files()
    print(f"\n📂 Found {len(long_audio_files)} long audio files:")
    for i, file in enumerate(long_audio_files[:5], 1):  # Show first 5
        print(f"   {i}. {os.path.basename(file)}")
    if len(long_audio_files) > 5:
        print(f"   ... and {len(long_audio_files) - 5} more")
