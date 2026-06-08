"""
Primate Vocalization Detection Pipeline

Automated detection of primate vocalizations (Cercopithecus nictitans and
Colobus guereza) in long rainforest audio recordings from Makokou, Gabon.

Four-class VGG19 temporal-frequency CRNN classifier (Cernic, Colobus_guereza,
Colobus_confuser, Background) with sliding-window detection, a low-frequency
spectral-energy gate, a three-filter false-positive cleanup pipeline, and
iterative hard-negative retraining.

Modules:
    config          All paths, hyperparameters, and species definitions.
    data_loader     Load and manage audio files.
    preprocessing   Convert waveforms to mel-spectrogram images.
    augmentation    Spectrogram-domain data augmentation (7x multiplier).
    model           VGG19 backbone + configurable pooling heads.
    train           End-to-end two-stage training pipeline.
    detection       Sliding-window detection with probability grouping and
                    the low-frequency spectral-energy gate.
    auto_cleanup    Three-filter automatic false-positive cleanup.
    utils           Visualization and analysis utilities.
"""

__version__ = "1.0.0"
__author__ = "Moshi Fu"

from . import config
from . import data_loader
from . import preprocessing
from . import augmentation
from . import model
from . import train
from . import detection
from . import auto_cleanup
from . import utils

__all__ = [
    'config',
    'data_loader',
    'preprocessing',
    'augmentation',
    'model',
    'train',
    'detection',
    'auto_cleanup',
    'utils',
]
