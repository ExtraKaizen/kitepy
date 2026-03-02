"""
kitepy - Dead Simple Deep Learning Framework

Structure:
- kitepy.core: Shared infrastructure (Engine, Config, etc.)
- kitepy.pillars: Domain-specific modules (Vision, Language, etc.)
"""

__version__ = "0.1.0"

# Import Core Components
from kitepy.core import (
    # Configs
    BaseConfig, TrainingConfig, DataConfig, ModelConfig,
    CNNConfig, TransformerConfig, LLMConfig, VLMConfig, RNNConfig,
    # Engine & Training
    Engine,
    Callback, CallbackList, EarlyStopping, ModelCheckpoint, 
    LRMonitor, ProgressLogger, LambdaCallback,
    # Metrics
    MetricTracker, accuracy, precision_recall_f1,
    # Base
    BaseModel,
    # Utils
    LRFinder, print_device_info, print_version_info, set_seed
)

# Import Registry
from kitepy.core.presets import list_models, list_datasets, register_preset

# Import Pillars
from kitepy.pillars.vision import (
    CNN, Transformer, Vision, 
    load_vision_data, get_train_transforms,
    DATASET_PRESETS, MODEL_PRESETS
)

# Forward compatibility aliases
load_data = load_vision_data

__all__ = [
    # Core
    'BaseConfig', 'TrainingConfig', 'DataConfig', 'ModelConfig',
    'CNNConfig', 'TransformerConfig', 'LLMConfig', 'VLMConfig', 'RNNConfig',
    'Engine', 'BaseModel', 'LRFinder',
    'Callback', 'CallbackList', 'EarlyStopping', 'ModelCheckpoint', 
    'LRMonitor', 'ProgressLogger', 'LambdaCallback',
    'MetricTracker', 'accuracy', 'precision_recall_f1',
    
    # Registry
    'list_models', 'list_datasets', 'register_preset',
    
    # Utils
    'print_device_info', 'print_version_info', 'set_seed',
    
    # Vision Pillar
    'CNN', 'Transformer', 'Vision',
    'load_vision_data', 'load_data', 'get_train_transforms',
    'DATASET_PRESETS', 'MODEL_PRESETS',
]