"""
Core shared infrastructure for kitepy.

This module contains all shared components that work across pillars:
- Engine: Universal training loop
- Config: All configuration classes
- Callbacks: Training callbacks
- Metrics: Metric tracking
- Utils: Utilities
- Base: Base model class
"""

from kitepy.core.config import (
    BaseConfig,
    TrainingConfig,
    DataConfig,
    ModelConfig,
    CNNConfig,
    TransformerConfig,
    LLMConfig,
    VLMConfig,
    RNNConfig,
)

from kitepy.core.engine import Engine
from kitepy.core.callbacks import (
    Callback,
    CallbackList,
    EarlyStopping,
    ModelCheckpoint,
    LRMonitor,
    ProgressLogger,
    LambdaCallback,
)
from kitepy.core.metrics import MetricTracker, accuracy, precision_recall_f1
from kitepy.core.base import BaseModel
from kitepy.core.lr_finder import LRFinder
from kitepy.core.utils import print_device_info, print_version_info, set_seed

__all__ = [
    # Configs
    'BaseConfig', 'TrainingConfig', 'DataConfig', 'ModelConfig',
    'CNNConfig', 'TransformerConfig', 'LLMConfig', 'VLMConfig', 'RNNConfig',
    # Engine
    'Engine',
    # Callbacks
    'Callback', 'CallbackList', 'EarlyStopping', 'ModelCheckpoint',
    'LRMonitor', 'ProgressLogger', 'LambdaCallback',
    # Metrics
    'MetricTracker', 'accuracy', 'precision_recall_f1',
    # Base
    'BaseModel',
    # LR Finder
    'LRFinder',
    # Utils
    'print_device_info', 'print_version_info', 'set_seed',
]
