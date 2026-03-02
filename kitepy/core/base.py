"""
Base model class with shared logic for all modalities.

All public APIs (CNN, LLM, VLM, etc.) inherit from this.
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Dict, Any, Callable
from pathlib import Path
import json

from .config import (
    BaseConfig,
    TrainingConfig,
    DataConfig,
    ModelConfig,
    merge_configs,
    validate_config,
)
# TODO: Decouple this from vision
from .presets import get_preset
from .utils import get_device, set_seed, print_model_summary


class BaseModel:
    """
    Base class for all model APIs.
    
    This provides:
    - .train() - Training
    - .evaluate() - Evaluation
    - .predict() - Inference
    - .save() / .load() - Persistence
    - .describe() - Inspection
    """
    
    # Subclasses must define these
    modality: str = None  # "vision", "language", "multimodal", etc.
    default_config_class = ModelConfig
    
    def __init__(
        self,
        model: Optional[Union[str, nn.Module]] = None,
        config: Optional[Union[Dict, BaseConfig, str]] = None,
        **kwargs
    ):
        """
        Initialize model.
        
        Args:
            model: Model name (preset), custom PyTorch model, or None
            config: Config dict, config object, or path to yaml/json
            **kwargs: Override any config values
        """
        self.model_name = model if isinstance(model, str) else "custom"
        self.custom_model = model if isinstance(model, nn.Module) else None
        
        # Build configuration
        self.config = self._build_config(model, config, **kwargs)
        
        # Initialize model (deferred to subclasses)
        self.model = None
        self.device = None
        self.is_trained = False
        
        # Training state
        self.training_history = []
        self.best_metric = None
        self.checkpoint_dir = None
    
    def _build_config(
        self,
        model: Optional[str],
        config: Optional[Union[Dict, BaseConfig, str]],
        **kwargs
    ) -> BaseConfig:
        """Build configuration from all sources."""
        # Start with default config
        default = self.default_config_class()
        
        # Get preset if model name provided
        preset = None
        if isinstance(model, str):
            try:
                preset = get_preset(model, self.modality)
            except ValueError:
                # Not a preset, maybe a model identifier for wrapper
                pass
        
        # Load user config if provided
        user_config = None
        if config is not None:
            if isinstance(config, str):
                # Load from file
                if config.endswith('.yaml') or config.endswith('.yml'):
                    user_config = self.default_config_class.from_yaml(config)
                elif config.endswith('.json'):
                    user_config = self.default_config_class.from_json(config)
                else:
                    raise ValueError(f"Unknown config file format: {config}")
            elif isinstance(config, dict):
                user_config = config
            elif isinstance(config, BaseConfig):
                user_config = config
        
        # Merge all configs
        final_config = merge_configs(default, preset, user_config, **kwargs)
        
        # Validate
        validate_config(final_config)
        
        return final_config
    
    def train(
        self,
        data: Optional[Union[str, Any]] = None,
        epochs: Optional[int] = None,
        **kwargs
    ):
        """
        Train the model.
        
        Args:
            data: Dataset name, path, or DataLoader
            epochs: Number of epochs (overrides config)
            **kwargs: Override any training config values
        
        Returns:
            self (for chaining)
        """
        # Import here to avoid circular dependency
        from .engine import Engine
        # TODO: Dynamic loading based on modality
        from kitepy.pillars.vision.data import load_data
        
        # Build training config
        train_config = TrainingConfig()
        if epochs is not None:
            kwargs['epochs'] = epochs
        train_config = merge_configs(train_config, user_config=None, **kwargs)
        
        # Auto-detect and adjust config for CPU/GPU (BEFORE data loading)
        if not torch.cuda.is_available():
            train_config.mixed_precision = False
            train_config.pin_memory = False
        
        # Build data config
        data_config = DataConfig()
        if data is not None:
            if isinstance(data, str):
                data_config.train_data = data
            # else assume it's already a DataLoader
        
        # Set seed for reproducibility
        set_seed(train_config.seed)
        
        # Initialize model if not done
        if self.model is None:
            self._build_model()
        
        # Enable gradient checkpointing if requested
        if hasattr(train_config, 'gradient_checkpointing') and train_config.gradient_checkpointing:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                print("✓ Gradient checkpointing enabled")
        
        # Move to device
        if self.device is None:
            self.device = get_device(train_config.devices)
        self.model = self.model.to(self.device)
        
        # Load data
        if isinstance(data, str) or data is None:
            train_loader, val_loader, _ = load_data(
                data if data else data_config.train_data,
                self.modality,
                data_config,
                train_config
            )
        else:
            # Assume data is already a DataLoader
            train_loader = data
            val_loader = None
        
        # Create engine and train
        engine = Engine(
            model=self.model,
            config=train_config,
            modality=self.modality
        )
        
        self.model, history = engine.train(
            train_loader=train_loader,
            val_loader=val_loader
        )
        
        self.training_history.extend(history)
        self.is_trained = True
        
        print(f"\n✓ Training complete! Best metric: {history[-1].get('val_loss', 'N/A')}")
        
        return self
    
    def evaluate(
        self,
        data: Union[str, Any],
        **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            data: Dataset name, path, or DataLoader
            **kwargs: Additional evaluation config
        
        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call .train() first or load a checkpoint.")
        
        from .engine import Engine
        from kitepy.pillars.vision.data import load_data
        
        # Load data
        if isinstance(data, str):
            data_config = DataConfig(test_data=data)
            train_config = TrainingConfig()
            _, _, test_loader = load_data(data, self.modality, data_config, train_config)
        else:
            test_loader = data
        
        # Create engine
        train_config = TrainingConfig()
        engine = Engine(
            model=self.model,
            config=train_config,
            modality=self.modality
        )
        
        # Evaluate
        metrics = engine.evaluate(test_loader)
        
        return metrics
    
    def predict(
        self,
        input: Union[torch.Tensor, Any],
        **kwargs
    ) -> torch.Tensor:
        """
        Run inference on input.
        
        Args:
            input: Input tensor or data
            **kwargs: Additional inference config
        
        Returns:
            Model output
        """
        if self.model is None:
            raise RuntimeError("Model not initialized.")
        
        self.model.eval()
        
        # Convert input to tensor if needed
        if not isinstance(input, torch.Tensor):
            input = self._preprocess_input(input)
        
        # Move to device
        if self.device is not None:
            input = input.to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(input)
        
        return output
    
    def save(self, path: str, save_config: bool = True):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            save_config: Whether to save config alongside model
        """
        if self.model is None:
            raise RuntimeError("No model to save.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'config': self.config.to_dict(),
            'is_trained': self.is_trained,
            'training_history': self.training_history,
        }
        
        torch.save(checkpoint, path)
        print(f"✓ Model saved to {path}")
        
        # Save config separately if requested
        if save_config:
            config_path = path.with_suffix('.yaml')
            self.config.to_yaml(config_path)
            print(f"✓ Config saved to {config_path}")
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None):
        """
        Load model from checkpoint.
        
        Args:
            path: Path to checkpoint
            device: Device to load model to
        
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(path, map_location='cpu')
        
        # Create instance
        instance = cls(
            model=checkpoint['model_name'],
            config=checkpoint['config']
        )
        
        # Build model and load weights
        instance._build_model()
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore state
        instance.is_trained = checkpoint.get('is_trained', True)
        instance.training_history = checkpoint.get('training_history', [])
        
        # Move to device
        if device:
            instance.device = device
            instance.model = instance.model.to(device)
        
        print(f"✓ Model loaded from {path}")
        
        return instance
    
    def describe(self):
        """Print model architecture and configuration."""
        print(f"\n{'='*70}")
        print(f"Model: {self.__class__.__name__}")
        print(f"Name: {self.model_name}")
        print(f"Modality: {self.modality}")
        print(f"Trained: {self.is_trained}")
        print(f"{'='*70}")
        
        print(f"\n📋 Configuration:")
        print(json.dumps(self.config.to_dict(), indent=2))
        
        if self.model is not None:
            print(f"\n🏗️  Architecture:")
            print_model_summary(self.model)
        else:
            print(f"\n⚠️  Model not initialized yet")
    
    def summary(self):
        """Print model parameter count and size."""
        if self.model is None:
            raise RuntimeError("Model not initialized.")
        
        print_model_summary(self.model)
    
    def explain_config(self):
        """Print all available configuration options."""
        print(f"\n{'='*70}")
        print(f"Configuration Options for {self.__class__.__name__}")
        print(f"{'='*70}\n")
        
        # Show default config with descriptions
        default_config = self.default_config_class()
        
        print("You can override any of these values:")
        print(json.dumps(default_config.to_dict(), indent=2))
        
        print(f"\n{'='*70}")
        print("Example usage:")
        print(f"{'='*70}")
        print(f"""
model = {self.__class__.__name__}(
    model="preset_name",
    config={{
        "lr": 0.001,
        "epochs": 50,
        # ... any config values
    }}
)

# Or override directly:
model = {self.__class__.__name__}(
    model="preset_name",
    lr=0.001,
    epochs=50
)
        """)
    
    def compile(self, mode: str = "default"):
        """
        Compile model for faster inference (PyTorch 2.0+).
        
        Args:
            mode: Compilation mode ("default", "reduce-overhead", "max-autotune")
        
        Returns:
            self (for chaining)
        """
        if self.model is None:
            raise RuntimeError("Model not initialized.")
        
        try:
            # Check PyTorch version
            torch_version = torch.__version__.split('+')[0]
            major, minor = map(int, torch_version.split('.')[:2])
            
            if major < 2:
                print(f"⚠️  torch.compile requires PyTorch 2.0+, you have {torch_version}")
                return self
            
            print(f"🔧 Compiling model with mode='{mode}'...")
            self.model = torch.compile(self.model, mode=mode)
            print("✓ Model compiled successfully")
        except Exception as e:
            print(f"⚠️  Compilation failed: {e}")
            print("Continuing with uncompiled model")
        
        return self
    
    def quantize(self, bits: int = 8):
        """
        Quantize model for faster inference.
        
        Args:
            bits: Quantization bits (4 or 8)
        
        Returns:
            self (for chaining)
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call _build_model() first.")
        
        try:
            if bits == 8:
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
                )
                print(f"✓ Model quantized to INT8")
            else:
                print(f"⚠️  {bits}-bit quantization requires specialized libraries")
                return self
        except Exception as e:
            print(f"⚠️  Quantization failed: {e}")
        
        return self
    
    def freeze_backbone(self):
        """
        Freeze all backbone layers (for transfer learning).
        
        Only the classifier/head will be trainable.
        
        Returns:
            self (for chaining)
        """
        if self.model is None:
            self._build_model()
        
        frozen_count = 0
        for name, param in self.model.named_parameters():
            # Keep classifier/head/fc trainable
            if any(k in name.lower() for k in ['classifier', 'head', 'fc', 'last']):
                param.requires_grad = True
            else:
                param.requires_grad = False
                frozen_count += 1
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        
        print(f"✓ Froze {frozen_count} layers")
        print(f"  - Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
        
        return self
    
    def unfreeze(self, layers: int = None):
        """
        Unfreeze layers for fine-tuning.
        
        Args:
            layers: Number of layers to unfreeze from the end (None = all)
        
        Returns:
            self (for chaining)
        """
        if self.model is None:
            raise RuntimeError("Model not initialized.")
        
        if layers is None:
            # Unfreeze all
            for param in self.model.parameters():
                param.requires_grad = True
            print("✓ Unfroze all layers")
        else:
            # Unfreeze last N layers
            params = list(self.model.named_parameters())
            for i, (name, param) in enumerate(reversed(params)):
                if i < layers:
                    param.requires_grad = True
            print(f"✓ Unfroze last {layers} layers")
        
        return self
    
    def profile(self, input_size: tuple = None):
        """
        Profile model: FLOPs, parameters, memory, and latency.
        
        Args:
            input_size: Input size tuple (batch, channels, height, width)
        
        Returns:
            Dict with profiling results
        """
        if self.model is None:
            self._build_model()
        
        # Default input size based on modality
        if input_size is None:
            input_size = (1, 3, 224, 224)
        
        print("\n" + "="*60)
        print("📊 Model Profile")
        print("="*60)
        
        # Parameter count
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\n📦 Parameters:")
        print(f"   Total:     {total_params:,}")
        print(f"   Trainable: {trainable_params:,}")
        
        # Model size in MB
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        total_size_mb = (param_size + buffer_size) / 1024 / 1024
        
        print(f"\n💾 Memory:")
        print(f"   Model size: {total_size_mb:.2f} MB")
        
        # Latency benchmark
        import time
        device = self.device if self.device else torch.device('cpu')
        self.model = self.model.to(device)
        self.model.eval()
        
        dummy_input = torch.randn(*input_size).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = self.model(dummy_input)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(10):
                start = time.perf_counter()
                _ = self.model(dummy_input)
                times.append(time.perf_counter() - start)
        
        avg_latency = sum(times) / len(times) * 1000
        throughput = 1000 / avg_latency * input_size[0]
        
        print(f"\n⚡ Performance (on {device}):")
        print(f"   Latency:    {avg_latency:.2f} ms")
        print(f"   Throughput: {throughput:.1f} samples/sec")
        
        print("\n" + "="*60)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_size_mb,
            'latency_ms': avg_latency,
            'throughput': throughput,
        }
    
    def export(self, path: str, format: str = "auto"):
        """
        Export model to ONNX or TorchScript.
        
        Args:
            path: Output file path
            format: "onnx", "torchscript", or "auto" (infer from extension)
        
        Returns:
            self (for chaining)
        """
        if self.model is None:
            raise RuntimeError("Model not initialized.")
        
        # Auto-detect format from extension
        if format == "auto":
            if path.endswith('.onnx'):
                format = "onnx"
            elif path.endswith('.pt') or path.endswith('.pth'):
                format = "torchscript"
            else:
                format = "torchscript"
        
        self.model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        
        if format == "onnx":
            try:
                torch.onnx.export(
                    self.model,
                    dummy_input,
                    path,
                    export_params=True,
                    opset_version=14,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                )
                print(f"✓ Exported to ONNX: {path}")
            except Exception as e:
                print(f"⚠️  ONNX export failed: {e}")
        
        elif format == "torchscript":
            try:
                traced = torch.jit.trace(self.model, dummy_input)
                traced.save(path)
                print(f"✓ Exported to TorchScript: {path}")
            except Exception as e:
                print(f"⚠️  TorchScript export failed: {e}")
        
        else:
            raise ValueError(f"Unknown format: {format}")
        
        return self
    
    def find_lr(self, data: str = "synthetic", **kwargs):
        """
        Find optimal learning rate using the LR Range Test.
        
        Args:
            data: Dataset name or "synthetic"
            **kwargs: Additional training arguments
        
        Returns:
            Suggested learning rate
        """
        from .lr_finder import LRFinder
        from kitepy.pillars.vision.data import load_data
        from .config import DataConfig, TrainingConfig, merge_configs
        
        if self.model is None:
            self._build_model()
        
        # Setup
        train_config = TrainingConfig()
        train_config = merge_configs(train_config, user_config=None, **kwargs)
        
        if not torch.cuda.is_available():
            train_config.pin_memory = False
        
        data_config = DataConfig()
        train_loader, _, _ = load_data(data, self.modality, data_config, train_config)
        
        device = get_device(train_config.devices)
        self.model = self.model.to(device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-7)
        criterion = nn.CrossEntropyLoss()
        
        # Run LR finder
        lr_finder = LRFinder(self.model, optimizer, criterion, device)
        lr_finder.find(train_loader)
        suggested_lr = lr_finder.suggest_lr()
        
        return suggested_lr
    
    def tune(self, data: str, trials: int = 20, **kwargs):
        """
        Hyperparameter tuning using Optuna.
        
        Args:
            data: Dataset name
            trials: Number of trials
            **kwargs: Fixed training arguments
        
        Returns:
            Best hyperparameters dict
        """
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna required. Install with: pip install optuna")
        
        def objective(trial):
            # Hyperparameters to tune
            lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
            
            # Create fresh model
            from copy import deepcopy
            model_copy = deepcopy(self)
            model_copy._build_model()
            
            # Train with these hyperparameters (short run for tuning)
            try:
                model_copy.train(
                    data=data,
                    epochs=kwargs.get('epochs', 3),
                    lr=lr,
                    batch_size=batch_size,
                    weight_decay=weight_decay,
                    fast_dev_run=False,
                    **{k: v for k, v in kwargs.items() if k not in ['epochs', 'lr', 'batch_size', 'weight_decay']}
                )
                
                # Return best validation loss
                if model_copy.training_history:
                    val_losses = [h.get('val_loss') for h in model_copy.training_history if h.get('val_loss')]
                    return min(val_losses) if val_losses else float('inf')
            except Exception as e:
                print(f"Trial failed: {e}")
                return float('inf')
            
            return float('inf')
        
        print(f"\n🔍 Starting hyperparameter search ({trials} trials)...")
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=trials, show_progress_bar=True)
        
        print(f"\n✓ Best hyperparameters found:")
        for key, value in study.best_params.items():
            print(f"   {key}: {value}")
        print(f"   Best val_loss: {study.best_value:.4f}")
        
        return study.best_params
    
    def unwrap(self) -> nn.Module:
        """
        Get the underlying PyTorch model.
        
        This is the escape hatch for users who want full PyTorch control.
        
        Returns:
            PyTorch nn.Module
        """
        if self.model is None:
            raise RuntimeError("Model not initialized.")
        
        return self.model
    
    # Abstract methods (must be implemented by subclasses)
    
    def _build_model(self):
        """Build the model. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _build_model()")
    
    def _preprocess_input(self, input: Any) -> torch.Tensor:
        """Preprocess input for inference. Can be overridden by subclasses."""
        return input
    
    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model_name}, trained={self.is_trained})"