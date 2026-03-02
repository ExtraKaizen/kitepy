"""
Learning Rate Finder - Auto-tune learning rate.

Based on the paper "Cyclical Learning Rates for Training Neural Networks"
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class LRFinder:
    """
    Learning rate finder to automatically discover optimal learning rate.
    
    Usage:
        lr_finder = LRFinder(model, optimizer, criterion, device)
        lrs, losses = lr_finder.find(train_loader, min_lr=1e-7, max_lr=10)
        best_lr = lr_finder.suggest_lr()
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        # Save original state
        self.model_state = model.state_dict()
        self.optimizer_state = optimizer.state_dict()
        
        # Results
        self.lrs = []
        self.losses = []
    
    def find(
        self,
        train_loader: DataLoader,
        min_lr: float = 1e-7,
        max_lr: float = 10.0,
        num_steps: int = 100,
        beta: float = 0.98
    ) -> Tuple[list, list]:
        """
        Run learning rate finder.
        
        Args:
            train_loader: Training data loader
            min_lr: Minimum learning rate to try
            max_lr: Maximum learning rate to try
            num_steps: Number of steps to run
            beta: Smoothing factor for loss
        
        Returns:
            (learning_rates, losses)
        """
        self.model.train()
        
        # Generate learning rates (exponential)
        lr_schedule = np.geomspace(min_lr, max_lr, num_steps)
        
        # Track best loss
        best_loss = float('inf')
        avg_loss = 0.0
        
        print(f"\n🔍 Running LR Finder ({num_steps} steps)...")
        
        iterator = iter(train_loader)
        
        for i, lr in enumerate(lr_schedule):
            # Set learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # Get batch
            try:
                data, target = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                data, target = next(iterator)
            
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Compute smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta ** (i + 1))
            
            # Track
            self.lrs.append(lr)
            self.losses.append(smoothed_loss)
            
            # Stop if loss explodes
            if i > 10 and smoothed_loss > 4 * best_loss:
                print(f"  Stopping early at step {i} (loss exploded)")
                break
            
            # Track best loss
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Print progress
            if (i + 1) % 20 == 0:
                print(f"  Step {i+1}/{num_steps} | LR: {lr:.2e} | Loss: {smoothed_loss:.4f}")
        
        # Restore original state
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)
        
        print("✓ LR Finder complete")
        
        return self.lrs, self.losses
    
    def suggest_lr(self, skip_start: int = 10, skip_end: int = 5) -> float:
        """
        Suggest optimal learning rate.
        
        Args:
            skip_start: Skip first N points (often unstable)
            skip_end: Skip last N points (often exploding)
        
        Returns:
            Suggested learning rate
        """
        if len(self.losses) == 0:
            raise RuntimeError("Must run find() first")
        
        # Find steepest negative gradient
        losses = self.losses[skip_start:-skip_end if skip_end > 0 else None]
        lrs = self.lrs[skip_start:-skip_end if skip_end > 0 else None]
        
        # Compute gradients
        gradients = np.gradient(losses)
        
        # Find minimum gradient (steepest descent)
        min_grad_idx = np.argmin(gradients)
        
        # Suggested LR is slightly before the steepest point
        suggested_idx = max(0, min_grad_idx - 5)
        suggested_lr = lrs[suggested_idx]
        
        print(f"\n💡 Suggested learning rate: {suggested_lr:.2e}")
        
        return suggested_lr
    
    def plot(self, save_path: str = None):
        """Plot learning rate vs loss curve."""
        if plt is None:
            print("Warning: matplotlib not installed, skipping plot.")
            return

        if len(self.losses) == 0:
            raise RuntimeError("Must run find() first")
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.lrs, self.losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        plt.grid(True, alpha=0.3)
        
        # Mark suggested LR
        suggested_lr = self.suggest_lr()
        plt.axvline(suggested_lr, color='r', linestyle='--', label=f'Suggested: {suggested_lr:.2e}')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            print(f"✓ Plot saved to {save_path}")
        else:
            plt.show()


__all__ = ['LRFinder']