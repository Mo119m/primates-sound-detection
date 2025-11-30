# Example Data

This directory is intended for small example audio files to demonstrate the pipeline functionality. Due to file size constraints and data sharing agreements, actual audio recordings are not included in this repository.

## Obtaining Sample Data

To test the pipeline, you will need:

1. Short audio clips (5 seconds) of target primate species
2. Background noise recordings from the same environment
3. Long audio recordings (10-30 minutes) for detection testing

### Data Format

Audio files should be:
- Format: WAV
- Sample rate: 44100 Hz (preferred, though pipeline can resample)
- Duration: 5 seconds for training clips, variable for long audio
- Mono or stereo (pipeline converts to mono internally)

### Organizing Your Data

Place your data in Google Drive following this structure:

```
chimp-audio/
├── audio/
│   ├── [Species1_name]/
│   │   └── *.wav (5-second clips)
│   ├── [Species2_name]/
│   │   └── *.wav (5-second clips)
│   └── background noise Clips 5sec/
│       └── *.wav (5-second clips)
└── long_audio/
    └── *.wav (long recordings for detection)
```

### Data Requirements

Minimum recommended data for training:
- At least 50 clips per primate species
- At least 200 background/environmental sound clips
- At least one long audio file for testing detection

More data generally improves model performance. The pipeline includes data augmentation to maximize utility of available samples.

### Creating Synthetic Test Data

For initial testing without real recordings, you can create synthetic audio:

```python
import numpy as np
import soundfile as sf

# Generate 5-second test audio at 44100 Hz
duration = 5.0
sr = 44100
t = np.linspace(0, duration, int(sr * duration))

# Synthetic primate call (frequency sweep)
frequency_start = 500
frequency_end = 2000
frequency = np.linspace(frequency_start, frequency_end, len(t))
phase = 2 * np.pi * np.cumsum(frequency) / sr
synthetic_call = np.sin(phase) * np.exp(-t/2)

# Add some noise
noise = np.random.randn(len(t)) * 0.05
synthetic_audio = synthetic_call + noise

# Save
sf.write('synthetic_primate_call.wav', synthetic_audio, sr)
```

Note that synthetic data will not capture the complexity of real vocalizations and should only be used for initial pipeline testing.

## Data Citation

If using publicly available primate vocalization datasets, please cite the original sources appropriately. Examples of relevant databases:

- Macaulay Library (Cornell Lab of Ornithology)
- Animal Sound Archive (Museum für Naturkunde Berlin)
- Xeno-canto (community-contributed wildlife sounds)

Always verify licensing and usage permissions before using third-party data.

## Contribution

If you have permission to share sample audio files suitable for demonstration purposes, please contact the repository maintainers. Ideal contributions would be:

- Small file size (under 5 MB total)
- Creative Commons or public domain licensed
- Representative of typical use cases
- Properly documented with species and recording metadata
