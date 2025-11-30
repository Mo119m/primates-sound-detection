"""
Preprocessing Module
====================
Convert audio waveforms to mel-spectrograms suitable for VGG19 input
"""

import numpy as np
import librosa
import cv2
from typing import Tuple
import config


def audio_to_melspectrogram(audio: np.ndarray, 
                            sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """
    Convert audio waveform to mel-spectrogram
    
    Args:
        audio: Audio waveform (1D array)
        sr: Sample rate
    
    Returns:
        Mel-spectrogram as 2D array
    """
    # Compute mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
        fmin=config.FMIN,
        fmax=config.FMAX
    )
    
    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db


def normalize_spectrogram(spec: np.ndarray) -> np.ndarray:
    """
    Normalize spectrogram to [0, 255] range
    
    Args:
        spec: Input spectrogram
    
    Returns:
        Normalized spectrogram
    """
    spec_min = spec.min()
    spec_max = spec.max()
    
    if spec_max - spec_min > 0:
        spec_norm = (spec - spec_min) / (spec_max - spec_min) * 255
    else:
        spec_norm = np.zeros_like(spec)
    
    return spec_norm.astype(np.uint8)


def resize_spectrogram(spec: np.ndarray, 
                       target_height: int = config.IMG_HEIGHT,
                       target_width: int = config.IMG_WIDTH) -> np.ndarray:
    """
    Resize spectrogram to target dimensions
    
    Args:
        spec: Input spectrogram (2D)
        target_height: Target height
        target_width: Target width
    
    Returns:
        Resized spectrogram
    """
    return cv2.resize(spec, (target_width, target_height), 
                     interpolation=cv2.INTER_LINEAR)


def spectrogram_to_rgb(spec: np.ndarray) -> np.ndarray:
    """
    Convert single-channel spectrogram to 3-channel RGB image
    
    Args:
        spec: Input spectrogram (2D, normalized to 0-255)
    
    Returns:
        RGB image (H, W, 3)
    """
    # Stack the same spectrogram 3 times for RGB channels
    rgb_image = np.stack([spec, spec, spec], axis=-1)
    return rgb_image


def preprocess_audio(audio: np.ndarray, 
                     sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """
    Complete preprocessing pipeline: audio -> mel-spectrogram -> RGB image
    
    Args:
        audio: Audio waveform
        sr: Sample rate
    
    Returns:
        Preprocessed image ready for VGG19 (H, W, 3), normalized to [0, 255]
    """
    # Step 1: Convert to mel-spectrogram
    mel_spec = audio_to_melspectrogram(audio, sr)
    
    # Step 2: Normalize to [0, 255]
    mel_spec_norm = normalize_spectrogram(mel_spec)
    
    # Step 3: Resize to target dimensions
    mel_spec_resized = resize_spectrogram(mel_spec_norm)
    
    # Step 4: Convert to RGB
    rgb_image = spectrogram_to_rgb(mel_spec_resized)
    
    return rgb_image


def preprocess_for_model(image: np.ndarray) -> np.ndarray:
    """
    Final preprocessing for model input (VGG19 expects [0, 1] range)
    
    Args:
        image: RGB image in [0, 255] range
    
    Returns:
        Image normalized to [0, 1] range
    """
    return image.astype(np.float32) / 255.0


def batch_preprocess_audio(audio_list: list, 
                           sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """
    Preprocess a batch of audio samples
    
    Args:
        audio_list: List of audio waveforms
        sr: Sample rate
    
    Returns:
        Batch of preprocessed images (N, H, W, 3)
    """
    preprocessed = []
    
    for i, audio in enumerate(audio_list):
        try:
            img = preprocess_audio(audio, sr)
            preprocessed.append(img)
        except Exception as e:
            print(f"⚠️  Warning: Failed to preprocess audio {i}: {e}")
    
    return np.array(preprocessed)


def extract_sliding_windows(audio: np.ndarray, 
                           sr: int = config.SAMPLE_RATE,
                           window_size: float = config.WINDOW_SIZE,
                           stride: float = config.WINDOW_STRIDE) -> Tuple[list, list]:
    """
    Extract sliding windows from long audio for detection
    
    Args:
        audio: Long audio waveform
        sr: Sample rate
        window_size: Window size in seconds
        stride: Stride in seconds
    
    Returns:
        Tuple of (window_audios, window_times)
        - window_audios: List of audio segments
        - window_times: List of (start_time, end_time) tuples
    """
    window_samples = int(window_size * sr)
    stride_samples = int(stride * sr)
    
    windows = []
    times = []
    
    for start in range(0, len(audio) - window_samples + 1, stride_samples):
        end = start + window_samples
        
        window_audio = audio[start:end]
        start_time = start / sr
        end_time = end / sr
        
        windows.append(window_audio)
        times.append((start_time, end_time))
    
    return windows, times


def visualize_spectrogram(spec: np.ndarray, title: str = "Mel-Spectrogram"):
    """
    Visualize a spectrogram using matplotlib
    
    Args:
        spec: Spectrogram to visualize (2D array)
        title: Plot title
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 4))
    plt.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency Bins')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Test preprocessing
    print("Testing Preprocessing Module")
    
    # Generate a test audio signal (sine wave)
    duration = 5.0
    sr = config.SAMPLE_RATE
    t = np.linspace(0, duration, int(sr * duration))
    frequency = 440  # A4 note
    test_audio = np.sin(2 * np.pi * frequency * t)
    
    print(f"\n Test Audio:")
    print(f"   Duration: {duration}s")
    print(f"   Sample Rate: {sr} Hz")
    print(f"   Shape: {test_audio.shape}")
    
    # Test mel-spectrogram conversion
    mel_spec = audio_to_melspectrogram(test_audio, sr)
    print(f"\n Mel-Spectrogram:")
    print(f"   Shape: {mel_spec.shape}")
    print(f"   Range: [{mel_spec.min():.2f}, {mel_spec.max():.2f}] dB")
    
    # Test full preprocessing
    rgb_image = preprocess_audio(test_audio, sr)
    print(f"\n Preprocessed Image:")
    print(f"   Shape: {rgb_image.shape}")
    print(f"   Range: [{rgb_image.min()}, {rgb_image.max()}]")
    
    # Test model input format
    model_input = preprocess_for_model(rgb_image)
    print(f"\n Model Input:")
    print(f"   Shape: {model_input.shape}")
    print(f"   Range: [{model_input.min():.4f}, {model_input.max():.4f}]")
    
    print("\n Preprocessing module test completed!")
