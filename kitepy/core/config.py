"""
Configuration system for kitepy.

All models use these config dataclasses for:
- Strong defaults
- Type checking
- Easy serialization
- User overrides
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Union, Literal, Any, Dict, List
import yaml
import json


# ============================================================================
# BASE CONFIGS
# ============================================================================

@dataclass
class BaseConfig:
    """Base configuration class with serialization methods."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def to_yaml(self, path: str):
        """Save config to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def to_json(self, path: str):
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, path: str):
        """Load config from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, path: str):
        """Load config from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# ============================================================================
# TRAINING CONFIG
# ============================================================================

@dataclass
class TrainingConfig(BaseConfig):
    """Universal training configuration for all modalities.
    
    These settings work across CNN, LLM, VLM, etc.
    """
    
    # Training duration
    epochs: int = 10
    max_steps: Optional[int] = None  # Override epochs if set
    
    # Batch size
    batch_size: Union[int, str] = 32  # Can be "auto" for auto-tuning
    gradient_accumulation_steps: int = 1
    
    # Learning rate
    lr: Union[float, str] = 1e-3  # Can be "auto" for auto-scaling
    lr_find: bool = False  # Run LR finder before training
    lr_find_min: float = 1e-7
    lr_find_max: float = 10.0
    lr_find_num_steps: int = 100
    weight_decay: float = 0.01
    warmup_steps: Union[int, float] = 0  # Can be fraction (0.1 = 10% of total)
    
    # Optimizer
    optimizer: Literal["adam", "adamw", "sgd", "lion"] = "adamw"
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Scheduler
    scheduler: Literal["cosine", "linear", "constant", "polynomial", "none"] = "cosine"
    scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Regularization
    max_grad_norm: float = 1.0  # Gradient clipping
    dropout: float = 0.0
    label_smoothing: float = 0.0
    
    # Precision
    mixed_precision: Union[bool, str] = True  # True/"fp16", "bf16", "fp32"
    
    # Distributed training
    strategy: Literal["auto", "ddp", "fsdp", "deepspeed"] = "auto"
    num_nodes: int = 1
    devices: Union[int, str] = "auto"  # Number of GPUs or "auto"
    
    # Checkpointing
    save_every_n_epochs: Optional[int] = 1
    save_top_k: int = 3  # Keep best k checkpoints
    checkpoint_dir: str = "./checkpoints"
    resume_from_checkpoint: Optional[str] = None
    
    # Logging
    log_every_n_steps: int = 50
    val_check_interval: Union[int, float] = 1.0  # Validate every n steps or fraction
    
    # Early stopping
    early_stopping: bool = False
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-4
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = False
    
    # Performance
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    # Debugging
    fast_dev_run: bool = False  # Run 1 batch for debugging
    overfit_batches: Union[int, float] = 0  # Overfit on n batches
    detect_anomaly: bool = False  # Detect NaN/Inf
    
    # Gradient checkpointing (for large models)
    gradient_checkpointing: bool = False
    
    # Loss function
    loss: Union[str, Any] = "cross_entropy"  # "cross_entropy", "focal", "bce", "mse", or custom
    class_weights: Optional[List[float]] = None  # For imbalanced datasets
    
    # Augmentation
    mixup_alpha: float = 0.0  # MixUp alpha (0 = disabled)
    cutmix_alpha: float = 0.0  # CutMix alpha (0 = disabled)
    
    # Callbacks (list of callback names or objects)
    callbacks: Optional[List[Any]] = None


# ============================================================================
# DATA CONFIG
# ============================================================================

@dataclass
class DataConfig(BaseConfig):
    """Data loading and preprocessing configuration."""
    
    # Data source
    train_data: Optional[str] = None
    val_data: Optional[str] = None
    test_data: Optional[str] = None
    
    # Data splits (if single dataset)
    train_split: Union[str, float] = "train"  # Can be "train" or 0.8
    val_split: Union[str, float] = "validation"
    test_split: Union[str, float] = "test"
    
    # Preprocessing
    preprocessing: str = "auto"  # "auto", "none", or custom
    augmentation: str = "auto"  # "auto", "none", "light", "heavy"
    
    # Sampling
    shuffle: bool = True
    drop_last: bool = False
    
    # Streaming (for large datasets)
    streaming: bool = False
    buffer_size: int = 10000


# ============================================================================
# MODEL CONFIGS
# ============================================================================

@dataclass
class ModelConfig(BaseConfig):
    """Base model configuration."""
    
    # Pretrained weights
    pretrained: bool = False
    pretrained_path: Optional[str] = None
    
    # Model modifications
    freeze_backbone: bool = False
    freeze_embeddings: bool = False
    
    # Regularization
    dropout: float = 0.0
    drop_path: float = 0.0  # Stochastic depth


@dataclass
class CNNConfig(ModelConfig):
    """Configuration for CNN models (ResNet, VGG, EfficientNet, etc.)."""
    
    # Architecture
    arch: str = "resnet"  # Architecture family
    depth: int = 18  # Model depth (18, 34, 50, 101, etc.)
    width_multiplier: float = 1.0  # Width scaling
    
    # Task
    num_classes: int = 1000
    in_channels: int = 3
    
    # Architecture-specific
    groups: int = 1  # For ResNeXt
    base_width: int = 64  # For ResNeXt
    
    # Advanced
    global_pool: str = "avg"  # "avg", "max", "avgmax"
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0


@dataclass
class TransformerConfig(ModelConfig):
    """Configuration for Vision Transformers (ViT, Swin, DeiT, etc.)."""
    
    # Architecture
    arch: str = "vit"  # "vit", "swin", "deit", etc.
    
    # Image specs
    img_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    
    # Transformer specs
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    
    # Task
    num_classes: int = 1000
    
    # Regularization
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    
    # Position embedding
    pos_embed_type: str = "learned"  # "learned", "sincos"


@dataclass
class LLMConfig(ModelConfig):
    """Configuration for Language Models (GPT, LLaMA, etc.)."""
    
    # Architecture
    arch: str = "gpt"  # "gpt", "llama", "mistral", etc.
    
    # Model size
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    d_ff: int = 3072  # FFN inner dimension (usually 4 * d_model)
    
    # Vocabulary
    vocab_size: int = 50257
    context_length: int = 2048
    
    # Attention
    attention_type: str = "multihead"  # "multihead", "mqa", "gqa"
    n_kv_heads: Optional[int] = None  # For GQA
    
    # Positional encoding
    pos_encoding: str = "learned"  # "learned", "rope", "alibi", "sincos"
    
    # Normalization
    norm_type: str = "layernorm"  # "layernorm", "rmsnorm"
    norm_eps: float = 1e-5
    
    # Activation
    activation: str = "gelu"  # "gelu", "swiglu", "relu"
    
    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    
    # Special tokens
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    
    # Initialization
    initializer_range: float = 0.02
    
    # Advanced
    use_cache: bool = True
    tie_word_embeddings: bool = True
    rope_theta: float = 10000.0  # For RoPE


@dataclass
class VLMConfig(ModelConfig):
    """Configuration for Vision-Language Models (CLIP, BLIP, etc.)."""
    
    # Architecture
    arch: str = "clip"  # "clip", "blip", "flamingo", etc.
    
    # Vision encoder
    vision_model: str = "vit-base"
    vision_config: Optional[TransformerConfig] = None
    
    # Text encoder
    text_model: str = "gpt-base"
    text_config: Optional[LLMConfig] = None
    
    # Fusion
    fusion_type: str = "cross_attention"  # "cross_attention", "concat", "add"
    projection_dim: int = 512  # For contrastive learning
    
    # Task
    task: str = "contrastive"  # "contrastive", "captioning", "vqa"
    
    # Loss
    temperature: float = 0.07  # For contrastive loss


@dataclass
class RNNConfig(ModelConfig):
    """Configuration for RNN models (LSTM, GRU)."""
    
    # Architecture
    rnn_type: str = "lstm"  # "lstm", "gru", "rnn"
    
    # Model size
    input_size: int = 768
    hidden_size: int = 256
    num_layers: int = 2
    
    # Regularization
    dropout: float = 0.0
    
    # Bidirectional
    bidirectional: bool = False
    
    # Task
    num_classes: int = 1000


# ============================================================================
# COMBINED CONFIG
# ============================================================================

@dataclass
class ExperimentConfig(BaseConfig):
    """Complete experiment configuration combining all sub-configs."""
    
    # Sub-configs
    model: Union[CNNConfig, TransformerConfig, LLMConfig, VLMConfig, RNNConfig] = field(
        default_factory=CNNConfig
    )
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Experiment metadata
    experiment_name: str = "default_experiment"
    tags: list = field(default_factory=list)
    notes: str = ""
    
    # Tracking
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None


# ============================================================================
# CONFIG MERGING UTILITIES
# ============================================================================

def merge_configs(
    default: BaseConfig,
    preset: Optional[Dict[str, Any]] = None,
    user_config: Optional[Union[Dict[str, Any], BaseConfig]] = None,
    **kwargs
) -> BaseConfig:
    """
    Merge configurations with priority: kwargs > user_config > preset > default.
    
    Args:
        default: Default config object
        preset: Preset config dict (from registry)
        user_config: User-provided config (dict or config object)
        **kwargs: Direct overrides
    
    Returns:
        Merged config object
    """
    # Start with default
    config_dict = default.to_dict()
    
    # Apply preset
    if preset:
        config_dict.update(preset)
    
    # Apply user config
    if user_config:
        if isinstance(user_config, BaseConfig):
            user_config = user_config.to_dict()
        config_dict.update(user_config)
    
    # Apply kwargs (highest priority)
    config_dict.update(kwargs)
    
    # Return new config object
    return type(default).from_dict(config_dict)


def validate_config(config: BaseConfig) -> bool:
    """
    Validate configuration values.
    
    Raises:
        ValueError: If config is invalid
    """
    if isinstance(config, TrainingConfig):
        if config.epochs <= 0 and config.max_steps is None:
            raise ValueError("Either epochs or max_steps must be positive")
        
        if isinstance(config.batch_size, int) and config.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if isinstance(config.lr, float) and config.lr <= 0:
            raise ValueError("learning rate must be positive")
    
    return True


# ============================================================================
# AUTO-TUNING UTILITIES
# ============================================================================

def auto_batch_size(model_size: str, available_memory_gb: float) -> int:
    """Auto-tune batch size based on model size and available GPU memory."""
    # Rough heuristics (can be improved)
    size_to_batch = {
        "small": 128,
        "base": 64,
        "large": 32,
        "xlarge": 16,
    }
    
    # Adjust for available memory
    memory_factor = available_memory_gb / 24.0  # Normalize to 24GB GPU
    base_batch = size_to_batch.get(model_size, 32)
    
    return max(1, int(base_batch * memory_factor))


def auto_learning_rate(batch_size: int, base_lr: float = 1e-3) -> float:
    """Auto-scale learning rate with batch size (linear scaling rule)."""
    base_batch = 256
    return base_lr * (batch_size / base_batch)