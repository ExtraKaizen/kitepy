"""
Utility functions for kitepy.

Device detection, logging, model summary, reproducibility, etc.
"""

import torch
import torch.nn as nn
import random
import numpy as np
import os
from typing import Optional, Union
from pathlib import Path


# ============================================================================
# DEVICE UTILITIES
# ============================================================================

def get_device(device: Union[str, int] = "auto") -> torch.device:
    """
    Get the appropriate device for training/inference.
    
    Args:
        device: "auto", "cpu", "cuda", "mps", or GPU index (0, 1, etc.)
    
    Returns:
        torch.device
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")  # Apple Silicon
        else:
            return torch.device("cpu")
    
    if isinstance(device, int):
        if torch.cuda.is_available() and device < torch.cuda.device_count():
            return torch.device(f"cuda:{device}")
        else:
            raise ValueError(f"GPU {device} not available")
    
    return torch.device(device)


def get_num_gpus() -> int:
    """Get number of available GPUs."""
    return torch.cuda.device_count()


def get_gpu_memory(device: int = 0) -> float:
    """
    Get available GPU memory in GB.
    
    Args:
        device: GPU index
    
    Returns:
        Available memory in GB
    """
    if not torch.cuda.is_available():
        return 0.0
    
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info()
    return free / (1024 ** 3)  # Convert to GB


def print_device_info():
    """Print information about available devices."""
    print("\n" + "="*70)
    print("Device Information")
    print("="*70)
    
    # CPU
    print(f"CPU: Available")
    
    # CUDA
    if torch.cuda.is_available():
        print(f"CUDA: Available")
        print(f"  - GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024 ** 3)
            print(f"  - GPU {i}: {props.name} ({memory_gb:.1f} GB)")
    else:
        print(f"CUDA: Not available")
    
    # MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        print(f"MPS (Apple Silicon): Available")
    
    print("="*70 + "\n")


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed: int = 42, deterministic: bool = False):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
        deterministic: If True, makes PyTorch operations deterministic
                      (may reduce performance)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Note: This can significantly slow down training
        print("⚠️  Deterministic mode enabled - training may be slower")
    else:
        torch.backends.cudnn.benchmark = True  # Faster on fixed input sizes


# ============================================================================
# MODEL SUMMARY
# ============================================================================

def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        trainable_only: If True, only count trainable parameters
    
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model: nn.Module) -> float:
    """
    Get model size in MB.
    
    Args:
        model: PyTorch model
    
    Returns:
        Model size in MB
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb


def print_model_summary(model: nn.Module, input_size: Optional[tuple] = None):
    """
    Print detailed model summary.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (e.g., (3, 224, 224))
    """
    print("\n" + "="*70)
    print("Model Summary")
    print("="*70)
    
    # Parameter counts
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    non_trainable_params = total_params - trainable_params
    
    print(f"Total parameters:       {total_params:,}")
    print(f"Trainable parameters:   {trainable_params:,}")
    print(f"Non-trainable params:   {non_trainable_params:,}")
    
    # Model size
    size_mb = get_model_size_mb(model)
    print(f"Model size:             {size_mb:.2f} MB")
    
    # Architecture
    print(f"\nArchitecture:")
    print(model)
    
    print("="*70 + "\n")


# ============================================================================
# LOGGING
# ============================================================================

class Logger:
    """Simple logger for training progress."""
    
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    def log(self, message: str, level: str = "INFO"):
        """Log a message."""
        formatted = f"[{level}] {message}"
        print(formatted)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(formatted + '\n')
    
    def info(self, message: str):
        """Log info message."""
        self.log(message, "INFO")
    
    def warning(self, message: str):
        """Log warning message."""
        self.log(message, "WARNING")
    
    def error(self, message: str):
        """Log error message."""
        self.log(message, "ERROR")


# ============================================================================
# CHECKPOINT UTILITIES
# ============================================================================

def get_checkpoint_path(
    checkpoint_dir: str,
    model_name: str,
    epoch: int,
    metric_name: str = "loss",
    metric_value: float = 0.0
) -> str:
    """
    Generate checkpoint path with naming convention.
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        model_name: Model name
        epoch: Current epoch
        metric_name: Metric name (e.g., "loss", "accuracy")
        metric_value: Metric value
    
    Returns:
        Checkpoint path
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    filename = f"{model_name}_epoch{epoch}_{metric_name}{metric_value:.4f}.pt"
    return str(Path(checkpoint_dir) / filename)


def cleanup_old_checkpoints(
    checkpoint_dir: str,
    keep_top_k: int = 3,
    metric_name: str = "loss",
    lower_is_better: bool = True
):
    """
    Keep only top-k checkpoints based on metric.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_top_k: Number of checkpoints to keep
        metric_name: Metric name to sort by
        lower_is_better: If True, keep checkpoints with lowest metric
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return
    
    # Find all checkpoint files
    checkpoints = list(checkpoint_dir.glob(f"*_{metric_name}*.pt"))
    
    if len(checkpoints) <= keep_top_k:
        return
    
    # Extract metric values from filenames
    checkpoint_metrics = []
    for ckpt in checkpoints:
        try:
            # Extract metric value from filename
            metric_str = ckpt.stem.split(f"{metric_name}")[-1]
            metric_value = float(metric_str.split('_')[0])
            checkpoint_metrics.append((ckpt, metric_value))
        except:
            continue
    
    # Sort by metric
    checkpoint_metrics.sort(key=lambda x: x[1], reverse=not lower_is_better)
    
    # Delete old checkpoints
    for ckpt, _ in checkpoint_metrics[keep_top_k:]:
        ckpt.unlink()
        print(f"Deleted old checkpoint: {ckpt.name}")


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def print_training_header(config):
    """Print training configuration header."""
    print("\n" + "="*70)
    print("Training Configuration")
    print("="*70)
    print(f"Epochs:           {config.epochs}")
    print(f"Batch size:       {config.batch_size}")
    print(f"Learning rate:    {config.lr}")
    print(f"Optimizer:        {config.optimizer}")
    print(f"Scheduler:        {config.scheduler}")
    print(f"Mixed precision:  {config.mixed_precision}")
    print(f"Device:           {config.devices}")
    print("="*70 + "\n")


def print_epoch_summary(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_loss: Optional[float] = None,
    train_acc: Optional[float] = None,
    val_acc: Optional[float] = None,
    lr: Optional[float] = None,
    epoch_time: Optional[float] = None
):
    """Print summary for one epoch."""
    metrics = [f"Train Loss: {train_loss:.4f}"]
    
    if val_loss is not None:
        metrics.append(f"Val Loss: {val_loss:.4f}")
    
    if train_acc is not None:
        metrics.append(f"Train Acc: {train_acc:.2f}%")
    
    if val_acc is not None:
        metrics.append(f"Val Acc: {val_acc:.2f}%")
    
    if lr is not None:
        metrics.append(f"LR: {lr:.2e}")
    
    if epoch_time is not None:
        metrics.append(f"Time: {format_time(epoch_time)}")
    
    print(f"Epoch {epoch}/{total_epochs} | " + " | ".join(metrics))


# ============================================================================
# MEMORY UTILITIES
# ============================================================================

def get_memory_usage() -> dict:
    """Get current memory usage statistics."""
    stats = {
        'cpu_memory_mb': 0,
        'gpu_memory_mb': 0,
        'gpu_memory_reserved_mb': 0,
    }
    
    if torch.cuda.is_available():
        stats['gpu_memory_mb'] = torch.cuda.memory_allocated() / (1024 ** 2)
        stats['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / (1024 ** 2)
    
    return stats


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# CONFIG HELPERS
# ============================================================================

def resolve_auto_value(value: Union[str, int, float], resolver_fn) -> Union[int, float]:
    """
    Resolve "auto" values in config.
    
    Args:
        value: Config value (can be "auto")
        resolver_fn: Function to call if value is "auto"
    
    Returns:
        Resolved value
    """
    if isinstance(value, str) and value.lower() == "auto":
        return resolver_fn()
    return value


def validate_path(path: Union[str, Path], create: bool = True) -> Path:
    """
    Validate and optionally create a path.
    
    Args:
        path: Path to validate
        create: If True, create directory if it doesn't exist
    
    Returns:
        Path object
    """
    path = Path(path)
    
    if create and not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    
    return path


# ============================================================================
# VERSION INFO
# ============================================================================

def print_version_info():
    """Print library and dependency versions."""
    print("\n" + "="*70)
    print("Environment Information")
    print("="*70)
    print(f"Python:        {os.sys.version.split()[0]}")
    print(f"PyTorch:       {torch.__version__}")
    
    try:
        import timm
        print(f"timm:          {timm.__version__}")
    except ImportError:
        print(f"timm:          Not installed")
    
    try:
        import transformers
        print(f"transformers:  {transformers.__version__}")
    except ImportError:
        print(f"transformers:  Not installed")
    
    try:
        import pytorch_lightning
        print(f"Lightning:     {pytorch_lightning.__version__}")
    except ImportError:
        print(f"Lightning:     Not installed")
    
    print("="*70 + "\n")


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    # Device
    'get_device',
    'get_num_gpus',
    'get_gpu_memory',
    'print_device_info',
    
    # Reproducibility
    'set_seed',
    
    # Model summary
    'count_parameters',
    'get_model_size_mb',
    'print_model_summary',
    
    # Logging
    'Logger',
    
    # Checkpoints
    'get_checkpoint_path',
    'cleanup_old_checkpoints',
    
    # Training
    'format_time',
    'get_lr',
    'print_training_header',
    'print_epoch_summary',
    
    # Memory
    'get_memory_usage',
    'clear_gpu_memory',
    
    # Helpers
    'resolve_auto_value',
    'validate_path',
    'print_version_info',
]