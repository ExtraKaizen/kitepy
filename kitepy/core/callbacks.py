"""
Callback system for training hooks.

Provides extensible hooks for custom behavior during training.
"""

from typing import Dict, Any, Optional, List, Callable
import torch
import torch.nn as nn
from pathlib import Path


class Callback:
    """Base callback class. Override methods to customize training behavior."""
    
    def on_train_start(self, trainer: 'Engine', **kwargs):
        """Called at the start of training."""
        pass
    
    def on_train_end(self, trainer: 'Engine', **kwargs):
        """Called at the end of training."""
        pass
    
    def on_epoch_start(self, trainer: 'Engine', epoch: int, **kwargs):
        """Called at the start of each epoch."""
        pass
    
    def on_epoch_end(self, trainer: 'Engine', epoch: int, metrics: Dict[str, Any], **kwargs):
        """Called at the end of each epoch."""
        pass
    
    def on_batch_start(self, trainer: 'Engine', batch_idx: int, **kwargs):
        """Called at the start of each batch."""
        pass
    
    def on_batch_end(self, trainer: 'Engine', batch_idx: int, loss: float, **kwargs):
        """Called at the end of each batch."""
        pass
    
    def on_validation_start(self, trainer: 'Engine', **kwargs):
        """Called at the start of validation."""
        pass
    
    def on_validation_end(self, trainer: 'Engine', metrics: Dict[str, Any], **kwargs):
        """Called at the end of validation."""
        pass


class EarlyStopping(Callback):
    """
    Early stopping callback.
    
    Args:
        monitor: Metric to monitor (default: 'val_loss')
        patience: Number of epochs with no improvement before stopping
        min_delta: Minimum change to qualify as an improvement
        mode: 'min' or 'max' (whether lower or higher is better)
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 5,
        min_delta: float = 1e-4,
        mode: str = 'min'
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.should_stop = False
    
    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        current = metrics.get(self.monitor)
        if current is None:
            return
        
        if self.mode == 'min':
            improved = current < self.best_value - self.min_delta
        else:
            improved = current > self.best_value + self.min_delta
        
        if improved:
            self.best_value = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"\n⚠️  Early stopping triggered at epoch {epoch}")


class ModelCheckpoint(Callback):
    """
    Save model checkpoints.
    
    Args:
        filepath: Path pattern for saving (can include {epoch}, {val_loss}, etc.)
        monitor: Metric to monitor for saving best model
        save_best_only: Only save when monitored metric improves
        save_top_k: Keep only top k checkpoints
        mode: 'min' or 'max'
    """
    
    def __init__(
        self,
        filepath: str = "checkpoints/model_{epoch:02d}.pt",
        monitor: str = 'val_loss',
        save_best_only: bool = False,
        save_top_k: int = 3,
        mode: str = 'min'
    ):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_top_k = save_top_k
        self.mode = mode
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.saved_checkpoints = []
    
    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        current = metrics.get(self.monitor, metrics.get('train_loss'))
        
        if self.save_best_only:
            if self.mode == 'min':
                improved = current < self.best_value
            else:
                improved = current > self.best_value
            
            if not improved:
                return
            self.best_value = current
        
        # Format filepath
        path = self.filepath.format(
            epoch=epoch,
            val_loss=metrics.get('val_loss', 0),
            train_loss=metrics.get('train_loss', 0),
        )
        
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'metrics': metrics,
        }
        torch.save(checkpoint, path)
        self.saved_checkpoints.append(path)
        print(f"  → Checkpoint saved: {path}")
        
        # Cleanup old checkpoints
        if len(self.saved_checkpoints) > self.save_top_k:
            old_path = self.saved_checkpoints.pop(0)
            if Path(old_path).exists():
                Path(old_path).unlink()


class LRMonitor(Callback):
    """Log learning rate at each epoch."""
    
    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        lr = trainer.optimizer.param_groups[0]['lr']
        print(f"  → Learning rate: {lr:.2e}")


class ProgressLogger(Callback):
    """
    Log training progress.
    
    Args:
        log_every_n_steps: Log every N batches
    """
    
    def __init__(self, log_every_n_steps: int = 50):
        self.log_every_n_steps = log_every_n_steps
        self.running_loss = 0.0
        self.batch_count = 0
    
    def on_batch_end(self, trainer, batch_idx, loss, **kwargs):
        self.running_loss += loss
        self.batch_count += 1
        
        if (batch_idx + 1) % self.log_every_n_steps == 0:
            avg_loss = self.running_loss / self.batch_count
            print(f"  Step {batch_idx + 1}: Loss = {avg_loss:.4f}")
            self.running_loss = 0.0
            self.batch_count = 0


class WandbLogger(Callback):
    """
    Log to Weights & Biases.
    
    Args:
        project: WandB project name
        name: Run name
        config: Config dict to log
    """
    
    def __init__(
        self,
        project: str = "kitepy",
        name: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        self.project = project
        self.name = name
        self.config = config
        self._wandb = None
    
    def on_train_start(self, trainer, **kwargs):
        try:
            import wandb
            self._wandb = wandb
            wandb.init(
                project=self.project,
                name=self.name,
                config=self.config or trainer.config.to_dict(),
            )
            wandb.watch(trainer.model)
            print(f"✓ WandB initialized: {self.project}")
        except ImportError:
            print("⚠️  wandb not installed. Install with: pip install wandb")
    
    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        if self._wandb:
            self._wandb.log(metrics, step=epoch)
    
    def on_train_end(self, trainer, **kwargs):
        if self._wandb:
            self._wandb.finish()


class TensorBoardLogger(Callback):
    """
    Log to TensorBoard.
    
    Args:
        log_dir: Directory for TensorBoard logs
    """
    
    def __init__(self, log_dir: str = "runs"):
        self.log_dir = log_dir
        self._writer = None
    
    def on_train_start(self, trainer, **kwargs):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(self.log_dir)
            print(f"✓ TensorBoard logging to: {self.log_dir}")
        except ImportError:
            print("⚠️  TensorBoard not available")
    
    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        if self._writer:
            for name, value in metrics.items():
                if value is not None:
                    self._writer.add_scalar(name, value, epoch)
    
    def on_train_end(self, trainer, **kwargs):
        if self._writer:
            self._writer.close()


class LambdaCallback(Callback):
    """
    Create callback from lambda functions.
    
    Args:
        on_epoch_end: Function to call at epoch end
        on_batch_end: Function to call at batch end
        etc.
    """
    
    def __init__(
        self,
        on_train_start: Optional[Callable] = None,
        on_train_end: Optional[Callable] = None,
        on_epoch_start: Optional[Callable] = None,
        on_epoch_end: Optional[Callable] = None,
        on_batch_start: Optional[Callable] = None,
        on_batch_end: Optional[Callable] = None,
    ):
        self._on_train_start = on_train_start
        self._on_train_end = on_train_end
        self._on_epoch_start = on_epoch_start
        self._on_epoch_end = on_epoch_end
        self._on_batch_start = on_batch_start
        self._on_batch_end = on_batch_end
    
    def on_train_start(self, trainer, **kwargs):
        if self._on_train_start:
            self._on_train_start(trainer, **kwargs)
    
    def on_train_end(self, trainer, **kwargs):
        if self._on_train_end:
            self._on_train_end(trainer, **kwargs)
    
    def on_epoch_start(self, trainer, epoch, **kwargs):
        if self._on_epoch_start:
            self._on_epoch_start(trainer, epoch, **kwargs)
    
    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        if self._on_epoch_end:
            self._on_epoch_end(trainer, epoch, metrics, **kwargs)
    
    def on_batch_start(self, trainer, batch_idx, **kwargs):
        if self._on_batch_start:
            self._on_batch_start(trainer, batch_idx, **kwargs)
    
    def on_batch_end(self, trainer, batch_idx, loss, **kwargs):
        if self._on_batch_end:
            self._on_batch_end(trainer, batch_idx, loss, **kwargs)


class CallbackList:
    """Manage multiple callbacks."""
    
    def __init__(self, callbacks: List[Callback] = None):
        self.callbacks = callbacks or []
    
    def add(self, callback: Callback):
        self.callbacks.append(callback)
    
    def on_train_start(self, trainer, **kwargs):
        for cb in self.callbacks:
            cb.on_train_start(trainer, **kwargs)
    
    def on_train_end(self, trainer, **kwargs):
        for cb in self.callbacks:
            cb.on_train_end(trainer, **kwargs)
    
    def on_epoch_start(self, trainer, epoch, **kwargs):
        for cb in self.callbacks:
            cb.on_epoch_start(trainer, epoch, **kwargs)
    
    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        for cb in self.callbacks:
            cb.on_epoch_end(trainer, epoch, metrics, **kwargs)
    
    def on_batch_start(self, trainer, batch_idx, **kwargs):
        for cb in self.callbacks:
            cb.on_batch_start(trainer, batch_idx, **kwargs)
    
    def on_batch_end(self, trainer, batch_idx, loss, **kwargs):
        for cb in self.callbacks:
            cb.on_batch_end(trainer, batch_idx, loss, **kwargs)
    
    def should_stop(self) -> bool:
        """Check if any callback requests stopping."""
        for cb in self.callbacks:
            if hasattr(cb, 'should_stop') and cb.should_stop:
                return True
        return False


# Export all callbacks
__all__ = [
    'Callback',
    'EarlyStopping', 
    'ModelCheckpoint',
    'LRMonitor',
    'ProgressLogger',
    'WandbLogger',
    'TensorBoardLogger',
    'LambdaCallback',
    'CallbackList',
]
