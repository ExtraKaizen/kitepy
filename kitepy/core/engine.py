"""
Universal training engine for all modalities.

Wraps PyTorch Lightning for automatic device handling, distributed training, etc.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, List, Dict, Any, Tuple
import time
from pathlib import Path
from tqdm import tqdm

from .config import TrainingConfig
from .utils import (
    get_device,
    get_lr,
    format_time,
    print_training_header,
    print_epoch_summary,
    get_checkpoint_path,
    cleanup_old_checkpoints,
)


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification."""
    
    def __init__(self, weight=None, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')
    
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ============================================================================
# TRAINING ENGINE
# ============================================================================

class Engine:
    """
    Universal training engine.
    
    For Phase 1 (MVP), this is a simple PyTorch training loop.
    In Phase 2+, this will wrap PyTorch Lightning for advanced features.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        modality: str = "vision",
    ):
        """
        Initialize training engine.
        
        Args:
            model: PyTorch model
            config: Training configuration
            modality: Model modality ("vision", "language", etc.)
        """
        self.model = model
        self.config = config
        self.modality = modality
        
        # Auto-detect and adjust config for CPU/GPU
        if not torch.cuda.is_available():
            self.config.mixed_precision = False
            self.config.pin_memory = False
            
        if self.config.fast_dev_run:
            self.config.epochs = 1
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.history = []
        
        # Setup device
        self.device = get_device(config.devices)
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = None  # Will be created in train() after knowing total steps
        
        # Setup loss function
        self.criterion = self._create_loss_function()
        
        # Mixed precision
        self.scaler = None
        if config.mixed_precision:
            if torch.cuda.is_available():
                self.scaler = torch.amp.GradScaler('cuda')
            else:
                self.scaler = None
                self.config.mixed_precision = False
                print("ℹ️  Mixed precision disabled (no GPU detected)")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create optimizer from config.
        
        Supports: adam, adamw, sgd, rmsprop, adadelta, adagrad, lion, custom
        """
        optimizer = self.config.optimizer
        
        # If user passed an optimizer instance directly
        if isinstance(optimizer, torch.optim.Optimizer):
            return optimizer
        
        # If user passed a callable (optimizer class or function)
        if callable(optimizer) and not isinstance(optimizer, str):
            return optimizer(self.model.parameters(), lr=self.config.lr, **self.config.optimizer_kwargs)
        
        # String-based optimizer lookup
        optimizer_name = optimizer.lower()
        params = self.model.parameters()
        lr = self.config.lr
        wd = self.config.weight_decay
        kwargs = self.config.optimizer_kwargs
        
        optimizers = {
            "adam": lambda: torch.optim.Adam(params, lr=lr, weight_decay=wd, **kwargs),
            "adamw": lambda: torch.optim.AdamW(params, lr=lr, weight_decay=wd, **kwargs),
            "sgd": lambda: torch.optim.SGD(params, lr=lr, momentum=kwargs.pop('momentum', 0.9), weight_decay=wd, **kwargs),
            "rmsprop": lambda: torch.optim.RMSprop(params, lr=lr, weight_decay=wd, **kwargs),
            "adadelta": lambda: torch.optim.Adadelta(params, lr=lr, weight_decay=wd, **kwargs),
            "adagrad": lambda: torch.optim.Adagrad(params, lr=lr, weight_decay=wd, **kwargs),
            "adamax": lambda: torch.optim.Adamax(params, lr=lr, weight_decay=wd, **kwargs),
            "nadam": lambda: torch.optim.NAdam(params, lr=lr, weight_decay=wd, **kwargs),
            "radam": lambda: torch.optim.RAdam(params, lr=lr, weight_decay=wd, **kwargs),
        }
        
        # Try Lion if available
        if optimizer_name == "lion":
            try:
                from lion_pytorch import Lion
                return Lion(params, lr=lr, weight_decay=wd, **kwargs)
            except ImportError:
                raise ImportError("Lion optimizer requires: pip install lion-pytorch")
        
        if optimizer_name not in optimizers:
            available = list(optimizers.keys()) + ["lion"]
            raise ValueError(f"Unknown optimizer: {optimizer_name}. Available: {available}")
        
        return optimizers[optimizer_name]()
    
    def _create_scheduler(self, total_steps: int):
        """
        Create learning rate scheduler.
        
        Supports: cosine, linear, step, exponential, polynomial, onecycle, warmup_cosine, custom
        """
        from torch.optim import lr_scheduler
        
        scheduler = self.config.scheduler
        
        # If user passed a scheduler instance directly
        if isinstance(scheduler, lr_scheduler.LRScheduler):
            return scheduler
        
        # If user passed a callable
        if callable(scheduler) and not isinstance(scheduler, str):
            return scheduler(self.optimizer, **self.config.scheduler_kwargs)
        
        scheduler_name = scheduler.lower()
        kwargs = self.config.scheduler_kwargs.copy()
        
        if scheduler_name == "none" or scheduler_name == "constant":
            return None
        
        elif scheduler_name == "cosine":
            return lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                **kwargs
            )
        
        elif scheduler_name == "linear":
            return lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=kwargs.pop('end_factor', 0.0),
                total_iters=total_steps,
                **kwargs
            )
        
        elif scheduler_name == "step":
            step_size = kwargs.pop('step_size', total_steps // 3)
            gamma = kwargs.pop('gamma', 0.1)
            return lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma, **kwargs)
        
        elif scheduler_name == "multistep":
            milestones = kwargs.pop('milestones', [total_steps // 3, 2 * total_steps // 3])
            gamma = kwargs.pop('gamma', 0.1)
            return lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma, **kwargs)
        
        elif scheduler_name == "exponential":
            gamma = kwargs.pop('gamma', 0.95)
            return lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma, **kwargs)
        
        elif scheduler_name == "polynomial":
            power = kwargs.pop('power', 1.0)
            return lr_scheduler.PolynomialLR(self.optimizer, total_iters=total_steps, power=power, **kwargs)
        
        elif scheduler_name == "onecycle":
            return lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.lr,
                total_steps=total_steps,
                **kwargs
            )
        
        elif scheduler_name == "warmup_cosine" or scheduler_name == "cosine_warmup":
            warmup_steps = kwargs.pop('warmup_steps', int(0.1 * total_steps))
            
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
            
            return lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        elif scheduler_name == "plateau":
            patience = kwargs.pop('patience', 10)
            factor = kwargs.pop('factor', 0.1)
            return lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=patience, factor=factor, **kwargs)
        
        else:
            available = ["none", "constant", "cosine", "linear", "step", "multistep", 
                        "exponential", "polynomial", "onecycle", "warmup_cosine", "plateau"]
            raise ValueError(f"Unknown scheduler: {scheduler_name}. Available: {available}")
    
    def _create_loss_function(self, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
        """
        Create loss function.
        
        Supports: cross_entropy, focal, bce, mse, mae, kl_div, custom
        Also supports class_weights for imbalanced datasets.
        """
        loss = getattr(self.config, 'loss', 'cross_entropy')
        
        # If user passed a loss function directly
        if isinstance(loss, nn.Module):
            return loss
        
        # If user passed a callable
        if callable(loss) and not isinstance(loss, str):
            return loss
        
        loss_name = loss.lower() if isinstance(loss, str) else 'cross_entropy'
        
        # Handle class weights
        weight = class_weights
        if weight is not None:
            weight = weight.to(self.device)
        
        label_smoothing = getattr(self.config, 'label_smoothing', 0.0)
        
        if loss_name == "cross_entropy" or loss_name == "ce":
            return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
        
        elif loss_name == "focal":
            # Focal Loss for imbalanced datasets
            return FocalLoss(weight=weight, gamma=2.0)
        
        elif loss_name == "bce" or loss_name == "binary_cross_entropy":
            return nn.BCEWithLogitsLoss(weight=weight)
        
        elif loss_name == "mse" or loss_name == "l2":
            return nn.MSELoss()
        
        elif loss_name == "mae" or loss_name == "l1":
            return nn.L1Loss()
        
        elif loss_name == "smooth_l1":
            return nn.SmoothL1Loss()
        
        elif loss_name == "kl_div" or loss_name == "kld":
            return nn.KLDivLoss(reduction='batchmean')
        
        elif loss_name == "nll":
            return nn.NLLLoss(weight=weight)
        
        elif loss_name == "margin":
            return nn.MultiMarginLoss()
        
        else:
            available = ["cross_entropy", "focal", "bce", "mse", "mae", "smooth_l1", "kl_div", "nll", "margin"]
            raise ValueError(f"Unknown loss: {loss_name}. Available: {available}")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Tuple[nn.Module, List[Dict[str, Any]]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
        
        Returns:
            (trained_model, training_history)
        """
        print_training_header(self.config)
        
        # Calculate total steps
        steps_per_epoch = len(train_loader)
        total_steps = self.config.epochs * steps_per_epoch
        
        # Create scheduler now that we know total steps
        self.scheduler = self._create_scheduler(total_steps)
        
        # Training loop
        for epoch in range(1, self.config.epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train one epoch
            train_loss, train_acc = self._train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = None, None
            if val_loader is not None:
                val_loss, val_acc = self._validate(val_loader)
            
            # Update history
            epoch_time = time.time() - epoch_start
            history_entry = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': get_lr(self.optimizer),
                'time': epoch_time,
            }
            self.history.append(history_entry)
            
            # Print summary
            print_epoch_summary(
                epoch=epoch,
                total_epochs=self.config.epochs,
                train_loss=train_loss,
                val_loss=val_loss,
                train_acc=train_acc,
                val_acc=val_acc,
                lr=get_lr(self.optimizer),
                epoch_time=epoch_time,
            )
            
            # Save checkpoint
            if self.config.save_every_n_epochs and epoch % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(epoch, val_loss if val_loss else train_loss)
            
            # Early stopping
            if self.config.early_stopping and val_loss is not None:
                if val_loss < self.best_val_loss - self.config.early_stopping_min_delta:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter = getattr(self, 'patience_counter', 0) + 1
                    if self.patience_counter >= self.config.early_stopping_patience:
                        print(f"\nEarly stopping triggered after {epoch} epochs")
                        break
        
        return self.model, self.history
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Add progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}", leave=False)

        for batch_idx, (data, target) in enumerate(pbar):
            # Move to device
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with torch.amp.autocast(device_type='cuda', enabled=self.scaler is not None):
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
            else:
                # Standard training (no mixed precision)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                self.optimizer.zero_grad()
                loss.backward()
                
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                
                self.optimizer.step()
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            
            if self.modality == "vision":
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
            
            self.global_step += 1
            
            # Log progress
            if (batch_idx + 1) % self.config.log_every_n_steps == 0:
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            # Fast dev run: only one batch
            if self.config.fast_dev_run:
                print(f"\n  [Fast Dev Run] Stopping after 1 batch")
                break
        
        pbar.close()
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                
                if self.modality == "vision":
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on test set."""
        test_loss, test_acc = self._validate(test_loader)
        
        metrics = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
        }
        
        print(f"\nTest Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.2f}%")
        
        return metrics
    
    def _save_checkpoint(self, epoch: int, metric_value: float):
        """Save model checkpoint."""
        checkpoint_path = get_checkpoint_path(
            checkpoint_dir=self.config.checkpoint_dir,
            model_name="model",
            epoch=epoch,
            metric_name="loss",
            metric_value=metric_value,
        )
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config.to_dict(),
            'history': self.history,
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"  → Checkpoint saved: {checkpoint_path}")
        
        # Cleanup old checkpoints
        cleanup_old_checkpoints(
            checkpoint_dir=self.config.checkpoint_dir,
            keep_top_k=self.config.save_top_k,
            metric_name="loss",
            lower_is_better=True,
        )


# ============================================================================
# EXPORT
# ============================================================================

__all__ = ['Engine']