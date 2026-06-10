"""
Data Augmentation 
Implements data augmentation strategies for mel-spectrograms
Based on tropical-stethoscope methods adapted for primate vocalizations
"""

import numpy as np
import random
from typing import List, Tuple

try:
    from . import config
except ImportError:  # Allow running as a standalone script (e.g. in Colab)
    import config


def _mel_cutoff_row(cutoff_hz: float,
                    n_mels: int = config.N_MELS,
                    fmin: float = config.FMIN,
                    fmax: float = config.FMAX) -> int:
    """Mel-spectrogram row index whose centre frequency first reaches cutoff_hz.

    Rows *below* this index hold the low-frequency Colobus roar; rows *at or
    above* it hold the high-frequency band that the nuisance augmentation
    randomizes. Imported lazily so augmentation.py has no hard top-level
    dependency on librosa for the geometric augmentations.
    """
    import librosa
    freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)
    idx = int(np.searchsorted(freqs, cutoff_hz))
    return int(np.clip(idx, 1, n_mels - 1))


def highfreq_nuisance(spec: np.ndarray,
                      background_spec: np.ndarray,
                      cutoff_hz: float = None) -> np.ndarray:
    """Randomize the HIGH-frequency band of a spectrogram, keeping the low band.

    Replaces everything above ``cutoff_hz`` with the high band of a random
    background clip while leaving the low-frequency content untouched. Both
    inputs are dB mel-spectrograms (freq = rows, row 0 = lowest frequency), so
    the high band is copied across directly with no domain conversion.

    Purpose (V12): the curated Colobus reference clips carry incidental
    high-frequency bird/insect energy that correlates with the Colobus label, so
    the model learned to fire on high-frequency texture and confused forest
    insects/birds with Colobus in the field. Swapping in *varied* high-frequency
    content across training examples decorrelates the high band from the label,
    forcing the model to rely on the invariant low-frequency roar -- the true
    Colobus signature. Applied ONLY to the Colobus class (Cernic IS high-freq and
    must keep its high band intact).
    """
    if cutoff_hz is None:
        cutoff_hz = config.COLOBUS_HF_CUTOFF_HZ
    if background_spec.shape != spec.shape:
        background_spec = random_crop_spectrogram(background_spec, spec.shape)
    row = _mel_cutoff_row(cutoff_hz, n_mels=spec.shape[0])
    out = spec.copy()
    out[row:, :] = background_spec[row:, :]
    return out


def add_background_noise(spec: np.ndarray, 
                        background_spec: np.ndarray,
                        snr_db: float = None) -> np.ndarray:
    """
    Mix a spectrogram with background noise at a given SNR
    
    Args:
        spec: Target spectrogram (signal)
        background_spec: Background spectrogram (noise)
        snr_db: Signal-to-noise ratio in dB (if None, random from config range)
    
    Returns:
        Mixed spectrogram
    """
    # Match dimensions if needed
    if background_spec.shape != spec.shape:
        background_spec = random_crop_spectrogram(background_spec, spec.shape)
    
    # Normalize both spectrograms
    spec_norm = normalize_spectrogram_for_mixing(spec)
    bg_norm = normalize_spectrogram_for_mixing(background_spec)
    
    # Calculate mixing ratio based on SNR
    if snr_db is None:
        snr_db = random.uniform(*config.BG_MIX_SNR_RANGE)
    
    # Convert SNR to linear scale ratio
    snr_linear = 10 ** (snr_db / 20)
    
    # Mix: signal + noise/snr_linear
    mixed = spec_norm + bg_norm / snr_linear
    
    # Renormalize to original range
    mixed = normalize_spectrogram_for_mixing(mixed)
    
    return mixed


def normalize_spectrogram_for_mixing(spec: np.ndarray) -> np.ndarray:
    """
    Normalize spectrogram for mixing operations
    """
    spec_min = spec.min()
    spec_max = spec.max()
    
    if spec_max - spec_min > 0:
        return (spec - spec_min) / (spec_max - spec_min)
    else:
        return np.zeros_like(spec)


def random_crop_spectrogram(spec: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Randomly crop a spectrogram to target shape
    """
    height, width = spec.shape
    target_height, target_width = target_shape
    
    if height >= target_height and width >= target_width:
        # Random crop
        start_h = random.randint(0, height - target_height)
        start_w = random.randint(0, width - target_width)
        return spec[start_h:start_h+target_height, start_w:start_w+target_width]
    else:
        # Pad if necessary
        pad_h = max(0, target_height - height)
        pad_w = max(0, target_width - width)
        spec_padded = np.pad(spec, ((0, pad_h), (0, pad_w)), mode='constant')
        return spec_padded[:target_height, :target_width]


def time_chop(spec: np.ndarray, chop_fraction: float = None) -> np.ndarray:
    """
    Crop from left or right edge (time axis)

    Args:
        spec: Input spectrogram (freq, time)
        chop_fraction: Fraction to crop (if None, random from config range)

    Returns:
        Cropped spectrogram
    """
    if chop_fraction is None:
        chop_fraction = random.uniform(*config.CHOP_RANGE)

    height, width = spec.shape
    # Clamp to [1, width-1] so we never produce an empty array (``spec[:, :-0]``
    # returns an empty slice) and never crop the entire signal away.
    chop_amount = int(width * chop_fraction)
    chop_amount = max(1, min(chop_amount, width - 1))

    # Randomly choose left or right
    if random.random() > 0.5:
        # Chop from left
        return spec[:, chop_amount:]
    else:
        # Chop from right
        return spec[:, :-chop_amount]


def freq_chop(spec: np.ndarray, chop_fraction: float = None) -> np.ndarray:
    """
    Crop from top or bottom edge (frequency axis)

    Args:
        spec: Input spectrogram (freq, time)
        chop_fraction: Fraction to crop (if None, random from config range)

    Returns:
        Cropped spectrogram
    """
    if chop_fraction is None:
        chop_fraction = random.uniform(*config.CHOP_RANGE)

    height, width = spec.shape
    # Clamp to [1, height-1] so we never produce an empty array (``spec[:-0]``
    # returns an empty slice) and never crop the entire signal away.
    chop_amount = int(height * chop_fraction)
    chop_amount = max(1, min(chop_amount, height - 1))

    # Randomly choose top or bottom
    if random.random() > 0.5:
        # Chop from top
        return spec[chop_amount:, :]
    else:
        # Chop from bottom
        return spec[:-chop_amount, :]


def translate(spec: np.ndarray, shift_amount: int = None) -> np.ndarray:
    """
    Shift spectrogram up or down (frequency translation)
    
    Args:
        spec: Input spectrogram (freq, time)
        shift_amount: Number of frequency bins to shift (if None, random from config range)
    
    Returns:
        Shifted spectrogram
    """
    if shift_amount is None:
        shift_amount = random.randint(*config.TRANSLATE_RANGE)
    
    if shift_amount > 0:
        # Shift up
        shifted = np.pad(spec, ((shift_amount, 0), (0, 0)), mode='constant')
        shifted = shifted[:-shift_amount, :]
    elif shift_amount < 0:
        # Shift down
        shift_amount = abs(shift_amount)
        shifted = np.pad(spec, ((0, shift_amount), (0, 0)), mode='constant')
        shifted = shifted[shift_amount:, :]
    else:
        shifted = spec.copy()
    
    return shifted


def resize_to_original_shape(spec: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
    """
    Resize augmented spectrogram back to original shape
    """
    import cv2
    target_height, target_width = original_shape
    return cv2.resize(spec, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def augment_spectrogram(spec: np.ndarray,
                       background_specs: List[np.ndarray] = None,
                       species_name: str = None) -> List[np.ndarray]:
    """
    Apply all augmentation strategies to a single spectrogram
    Following config.AUGMENTATION_CONFIG (Scheme A - Conservative)

    Args:
        spec: Input spectrogram
        background_specs: List of background spectrograms for noise mixing
        species_name: class label of this clip. When it matches
            config.COLOBUS_HF_AUG_CLASS, extra high-frequency-randomized variants
            are added (see highfreq_nuisance) so the model learns to key on the
            low-frequency Colobus roar rather than incidental high-freq texture.

    Returns:
        List of augmented spectrograms
    """
    augmented = []
    original_shape = spec.shape
    
    # Original version
    for _ in range(config.AUGMENTATION_CONFIG['original']):
        augmented.append(spec.copy())
    
    # Background noise mixing
    if background_specs is not None and len(background_specs) > 0:
        for _ in range(config.AUGMENTATION_CONFIG['background_noise_mix']):
            bg_spec = random.choice(background_specs)
            mixed = add_background_noise(spec, bg_spec)
            augmented.append(mixed)
    
    # Time chop
    for _ in range(config.AUGMENTATION_CONFIG['time_chop']):
        chopped = time_chop(spec.copy())
        chopped_resized = resize_to_original_shape(chopped, original_shape)
        augmented.append(chopped_resized)
    
    # Frequency chop
    for _ in range(config.AUGMENTATION_CONFIG['freq_chop']):
        chopped = freq_chop(spec.copy())
        chopped_resized = resize_to_original_shape(chopped, original_shape)
        augmented.append(chopped_resized)
    
    # Translate (frequency shift)
    for _ in range(config.AUGMENTATION_CONFIG['translate']):
        shifted = translate(spec.copy())
        augmented.append(shifted)

    # High-frequency nuisance randomization (Colobus only, V12).
    # Decorrelates the high band from the Colobus label so the model is forced to
    # rely on the low-frequency roar. Skipped for every other class (notably
    # Cernic, whose discriminative energy IS in the high band).
    if (species_name == config.COLOBUS_HF_AUG_CLASS
            and background_specs is not None and len(background_specs) > 0):
        for _ in range(config.COLOBUS_HF_AUG_COUNT):
            bg_spec = random.choice(background_specs)
            augmented.append(highfreq_nuisance(spec.copy(), bg_spec))

    return augmented


def augment_dataset(species_specs: dict,
                   background_specs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Augment entire dataset
    
    Args:
        species_specs: Dictionary mapping species name to list of spectrograms
        background_specs: List of background spectrograms
    
    Returns:
        Tuple of (X, y, sample_info)
        - X: Array of augmented spectrograms
        - y: Array of labels
        - sample_info: List of strings describing each sample
    """
    X_all = []
    y_all = []
    sample_info = []
    
    print("\n Augmenting Dataset")
    
    # Create label mapping
    label_map = {species: i for i, species in enumerate(config.CLASS_NAMES[:-1])}
    label_map['Background'] = len(label_map)
    
    # Augment each species
    for species_name, specs in species_specs.items():
        print(f"\n Augmenting {species_name}")
        species_label = label_map[species_name]

        species_aug_count = 0
        for i, spec in enumerate(specs):
            augmented_specs = augment_spectrogram(spec, background_specs,
                                                  species_name=species_name)

            for j, aug_spec in enumerate(augmented_specs):
                X_all.append(aug_spec)
                y_all.append(species_label)
                sample_info.append(f"{species_name}_sample{i}_aug{j}")
            species_aug_count += len(augmented_specs)

            if (i + 1) % 50 == 0:
                print(f"   Processed {i + 1}/{len(specs)} samples...")

        print(f"{len(specs)} → {species_aug_count} samples")
    
    # Add background (no augmentation)
    print(f"\n Adding Background samples")
    background_label = label_map['Background']
    for i, spec in enumerate(background_specs):
        X_all.append(spec)
        y_all.append(background_label)
        sample_info.append(f"Background_sample{i}")
    print(f"Added {len(background_specs)} background samples")
    
    # Convert to numpy arrays
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    
    print(f"\n Augmentation Complete")
    print(f"   Total samples: {len(X_all)}")
    print(f"   Shape: {X_all.shape}")
    
    return X_all, y_all, sample_info


if __name__ == "__main__":
    # Test augmentation
    print("Testing Augmentation Module")
    
    # Create dummy spectrogram
    test_spec = np.random.randn(128, 216)  # Typical mel-spec shape
    test_bg = np.random.randn(128, 216)
    
    print(f"\n Original Spectrogram Shape: {test_spec.shape}")
    
    # Test individual augmentations
    print("\n Testing Individual Augmentations:")
    
    mixed = add_background_noise(test_spec, test_bg)
    print(f"   Background Noise Mix: {mixed.shape}")
    
    chopped_time = time_chop(test_spec)
    print(f"   Time Chop: {chopped_time.shape}")
    
    chopped_freq = freq_chop(test_spec)
    print(f"   Freq Chop: {chopped_freq.shape}")

    shifted = translate(test_spec)
    print(f"   Translate: {shifted.shape}")
    
    # Test full augmentation
    print("\n Testing Full Augmentation Pipeline:")
    augmented = augment_spectrogram(test_spec, [test_bg])
    print(f"   Generated {len(augmented)} augmented versions")
    print(f"   Expected: {config.AUGMENTATION_MULTIPLIER}")
    
    print("\n Augmentation module test completed")
