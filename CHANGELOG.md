# Changelog

All notable changes to the Primate Vocalization Detection Pipeline will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-11

### Added

- Initial release of complete pipeline
- VGG19-based transfer learning model for primate vocalization classification
- Support for two species: Cercocebus torquatus and Colobus guereza
- Data loading module with automatic folder scanning
- Mel-spectrogram preprocessing pipeline
- Data augmentation strategies: background mixing, time/frequency cropping, frequency translation
- Training pipeline with class balancing and early stopping
- Sliding window detection for long audio files
- Non-maximum suppression for overlapping detections
- Comprehensive visualization tools for waveforms and spectrograms
- Hard negative mining workflow for domain gap reduction
- CSV export of detection results
- Audio clip extraction from detections
- Complete documentation suite
- Jupyter notebook workflow for Google Colab
- Modular configuration system
- MIT License

### Configuration Parameters

- Sample rate: 44100 Hz
- Mel-spectrogram: 128 bins, 2048 FFT, 512 hop length, 20-8000 Hz frequency range
- Augmentation multiplier: 7x for primate classes
- Model: VGG19 with frozen base layers, two dense layers with dropout
- Training: 50 epochs maximum, early stopping with patience 10
- Detection: 5-second windows, 2.5-second stride, confidence threshold 0.7

### Known Issues

- Domain gap between augmented training data and complex field recordings
- Model may overestimate primate calls in presence of bird vocalizations
- Class imbalance in detection results (Colobus detections outnumber Cercocebus approximately 17:1)

### Dependencies

- TensorFlow >= 2.10.0
- librosa >= 0.10.0
- soundfile >= 0.12.0
- scikit-learn >= 1.2.0
- pandas >= 1.5.0
- matplotlib >= 3.6.0
- numpy >= 1.23.0
- opencv-python >= 4.7.0

## [Unreleased]

### Planned Features

- Unit test suite
- Command-line interface for detection
- Support for additional species
- Model ensemble methods
- Spectrogram-based data augmentation (SpecAugment)
- Integration with bioacoustics annotation tools (Raven, Audacity)
- Batch processing optimizations
- Real-time detection capability
- Confidence calibration methods
- Active learning workflow for efficient labeling

### Under Consideration

- Alternative base architectures (ResNet, EfficientNet)
- Attention mechanisms for temporal modeling
- Multi-label classification for overlapping calls
- Unsupervised pre-training on unlabeled field recordings
- Integration with ecological metadata (time of day, weather conditions)
- Deployment as web service or mobile application
