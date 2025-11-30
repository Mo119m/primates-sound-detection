"""
Primate Vocalization Detection Pipeline

An automated system for detecting primate vocalizations in long-term rainforest 
audio recordings using deep learning with transfer learning from VGG19.

This package provides modules for:
- Audio data loading and preprocessing
- Mel-spectrogram conversion
- Data augmentation strategies
- VGG19-based classification model
- Training pipeline with class balancing
- Sliding window detection in long audio files
- Visualization and analysis tools

Author: Mo
Year: 2024
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Mo"

from . import config
from . import data_loader
from . import preprocessing
from . import augmentation
from . import model
from . import train
from . import detection
from . import utils

__all__ = [
    'config',
    'data_loader',
    'preprocessing',
    'augmentation',
    'model',
    'train',
    'detection',
    'utils',
]
