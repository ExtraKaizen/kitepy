"""
Comprehensive Examples: CNN & ViT

This file demonstrates ALL possible use cases of kitepy.
Each example uses fast_dev_run=True for quick testing.

Run with: python examples/quickstart_comprehensive_cnn_vit.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from kitepy import (
    CNN, Transformer,
    EarlyStopping, ModelCheckpoint, LRMonitor, LambdaCallback,
    MetricTracker, TrainingConfig, CNNConfig, TransformerConfig,
    print_device_info, print_version_info
)


# =============================================================================
# BASIC EXAMPLES
# =============================================================================

def example_1_simplest_cnn():
    """Example 1: The simplest possible CNN training."""
    print("\n" + "="*70)
    print("📌 Example 1: Simplest CNN Training (1 line!)")
    print("="*70)
    
    CNN("resnet18").train("synthetic", epochs=1, fast_dev_run=True, num_workers=0)
    print("✓ Done!")


def example_2_simplest_vit():
    """Example 2: The simplest possible ViT training."""
    print("\n" + "="*70)
    print("📌 Example 2: Simplest ViT Training")
    print("="*70)
    
    Transformer("vit_tiny").train("synthetic", epochs=1, fast_dev_run=True, num_workers=0)
    print("✓ Done!")


def example_3_with_config():
    """Example 3: Training with basic configuration."""
    print("\n" + "="*70)
    print("📌 Example 3: Training with Config")
    print("="*70)
    
    model = CNN("resnet18", num_classes=10)
    model.train(
        "synthetic",
        epochs=2,
        batch_size=32,
        lr=0.001,
        fast_dev_run=True,
        num_workers=0
    )
    print("✓ Done!")


# =============================================================================
# MODEL ARCHITECTURE EXAMPLES
# =============================================================================

def example_4_different_cnn_architectures():
    """Example 4: Different CNN architectures (just build, no training for speed)."""
    print("\n" + "="*70)
    print("📌 Example 4: Different CNN Architectures")
    print("="*70)
    
    # Just build models to test architecture support (fast)
    architectures = ["resnet18", "resnet50", "efficientnet_b0", "mobilenetv3_small_100"]
    
    for arch in architectures:
        print(f"\n→ Testing {arch}...")
        model = CNN(arch, num_classes=10)
        model._build_model()  # Just build, don't train (faster on CPU)
        print(f"  ✓ {arch}: {sum(p.numel() for p in model.model.parameters()):,} params")


def example_5_different_vit_architectures():
    """Example 5: Different ViT architectures."""
    print("\n" + "="*70)
    print("📌 Example 5: Different ViT Architectures")
    print("="*70)
    
    architectures = ["vit_tiny", "vit_small", "deit_tiny", "swin_tiny"]
    
    for arch in architectures:
        print(f"\n→ Testing {arch}...")
        model = Transformer(arch, num_classes=10)
        model._build_model()  # Just build, don't train all
        print(f"  ✓ {arch}: {sum(p.numel() for p in model.model.parameters()):,} params")


# =============================================================================
# OPTIMIZER EXAMPLES
# =============================================================================

def example_6_all_optimizers():
    """Example 6: All supported optimizers."""
    print("\n" + "="*70)
    print("📌 Example 6: All Optimizers")
    print("="*70)
    
    optimizers = ["adam", "adamw", "sgd", "rmsprop", "radam", "nadam", "adamax"]
    
    for opt in optimizers:
        print(f"\n→ Testing {opt}...")
        CNN("resnet18", num_classes=10).train(
            "synthetic",
            epochs=1,
            optimizer=opt,
            fast_dev_run=True,
            num_workers=0
        )
        print(f"  ✓ {opt} works!")


def example_7_custom_optimizer():
    """Example 7: Using a custom optimizer."""
    print("\n" + "="*70)
    print("📌 Example 7: Custom Optimizer")
    print("="*70)
    
    # Pass optimizer class directly
    CNN("resnet18", num_classes=10).train(
        "synthetic",
        epochs=1,
        optimizer=torch.optim.Adadelta,  # Any PyTorch optimizer
        fast_dev_run=True,
        num_workers=0
    )
    print("✓ Custom optimizer works!")


# =============================================================================
# SCHEDULER EXAMPLES
# =============================================================================

def example_8_all_schedulers():
    """Example 8: All supported schedulers."""
    print("\n" + "="*70)
    print("📌 Example 8: All Schedulers")
    print("="*70)
    
    schedulers = ["cosine", "linear", "step", "exponential", "polynomial", "onecycle"]
    
    for sched in schedulers:
        print(f"\n→ Testing {sched}...")
        CNN("resnet18", num_classes=10).train(
            "synthetic",
            epochs=1,
            scheduler=sched,
            fast_dev_run=True,
            num_workers=0
        )
        print(f"  ✓ {sched} works!")


def example_9_warmup_scheduler():
    """Example 9: Cosine with warmup."""
    print("\n" + "="*70)
    print("📌 Example 9: Warmup + Cosine Scheduler")
    print("="*70)
    
    CNN("resnet18", num_classes=10).train(
        "synthetic",
        epochs=1,
        scheduler="warmup_cosine",
        scheduler_kwargs={"warmup_steps": 100},
        fast_dev_run=True,
        num_workers=0
    )
    print("✓ Warmup cosine works!")


# =============================================================================
# LOSS FUNCTION EXAMPLES
# =============================================================================

def example_10_all_loss_functions():
    """Example 10: All supported loss functions."""
    print("\n" + "="*70)
    print("📌 Example 10: All Loss Functions")
    print("="*70)
    
    losses = ["cross_entropy", "focal"]
    
    for loss in losses:
        print(f"\n→ Testing {loss}...")
        CNN("resnet18", num_classes=10).train(
            "synthetic",
            epochs=1,
            loss=loss,
            fast_dev_run=True,
            num_workers=0
        )
        print(f"  ✓ {loss} works!")


def example_11_custom_loss():
    """Example 11: Custom loss function."""
    print("\n" + "="*70)
    print("📌 Example 11: Custom Loss Function")
    print("="*70)
    
    # Define custom loss
    class LabelSmoothingLoss(nn.Module):
        def __init__(self, smoothing=0.1):
            super().__init__()
            self.smoothing = smoothing
        
        def forward(self, pred, target):
            n_classes = pred.size(1)
            one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
            one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
            log_prob = torch.log_softmax(pred, dim=1)
            return -(one_hot * log_prob).sum(dim=1).mean()
    
    # Disable checkpointing for custom loss (local classes can't be pickled)
    CNN("resnet18", num_classes=10).train(
        "synthetic",
        epochs=1,
        loss=LabelSmoothingLoss(smoothing=0.1),
        fast_dev_run=True,
        num_workers=0,
        save_every_n_epochs=0  # Disable checkpointing
    )
    print("✓ Custom loss works!")


# =============================================================================
# AUGMENTATION EXAMPLES
# =============================================================================

def example_12_all_augmentations():
    """Example 12: All augmentation strategies."""
    print("\n" + "="*70)
    print("📌 Example 12: All Augmentation Strategies")
    print("="*70)
    
    # Note: For synthetic data, augmentation doesn't apply visually
    # but the transforms are created correctly
    from kitepy import get_train_transforms
    
    augmentations = ["none", "light", "heavy", "randaugment", "trivialaugment", "autoaugment"]
    
    for aug in augmentations:
        transform = get_train_transforms(224, aug)
        print(f"  ✓ {aug}: {len(transform.transforms)} transforms")


# =============================================================================
# TRANSFER LEARNING EXAMPLES
# =============================================================================

def example_13_pretrained_model():
    """Example 13: Using pretrained weights."""
    print("\n" + "="*70)
    print("📌 Example 13: Pretrained Model")
    print("="*70)
    
    model = CNN("resnet18", num_classes=10, pretrained=True)
    model._build_model()
    print(f"✓ Loaded pretrained ResNet18: {sum(p.numel() for p in model.model.parameters()):,} params")


def example_14_freeze_backbone():
    """Example 14: Freeze backbone for transfer learning."""
    print("\n" + "="*70)
    print("📌 Example 14: Freeze Backbone (Transfer Learning)")
    print("="*70)
    
    model = CNN("resnet50", num_classes=10, pretrained=True)
    model.freeze_backbone()
    
    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.model.parameters())
    
    print(f"✓ Frozen! Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


def example_15_gradual_unfreezing():
    """Example 15: Gradual unfreezing."""
    print("\n" + "="*70)
    print("📌 Example 15: Gradual Unfreezing")
    print("="*70)
    
    model = CNN("resnet18", num_classes=10)
    model.freeze_backbone()
    print("  → Backbone frozen")
    
    model.unfreeze(layers=10)
    print("  → Unfroze last 10 layers")
    
    model.unfreeze()
    print("  → Unfroze all layers")
    print("✓ Gradual unfreezing works!")


# =============================================================================
# CALLBACK EXAMPLES
# =============================================================================

def example_16_early_stopping():
    """Example 16: Early stopping callback."""
    print("\n" + "="*70)
    print("📌 Example 16: Early Stopping")
    print("="*70)
    
    es = EarlyStopping(patience=3, monitor='val_loss', mode='min')
    
    # Simulate epochs
    for epoch, val_loss in enumerate([0.5, 0.4, 0.45, 0.46, 0.47]):
        es.on_epoch_end(None, epoch, {'val_loss': val_loss})
        if es.should_stop:
            print(f"  → Would stop at epoch {epoch}")
            break
    
    print("✓ Early stopping works!")


def example_17_model_checkpoint():
    """Example 17: Model checkpoint callback."""
    print("\n" + "="*70)
    print("📌 Example 17: Model Checkpoint")
    print("="*70)
    
    mc = ModelCheckpoint(
        filepath="checkpoints/best_model.pt",
        save_best_only=True,
        monitor='val_loss'
    )
    print("✓ ModelCheckpoint created")
    print(f"  → Save to: {mc.filepath}")
    print(f"  → Save best only: {mc.save_best_only}")


def example_18_lambda_callback():
    """Example 18: Custom lambda callback."""
    print("\n" + "="*70)
    print("📌 Example 18: Lambda Callback (Custom Hooks)")
    print("="*70)
    
    epoch_losses = []
    
    def log_loss(trainer, epoch, metrics, **kwargs):
        epoch_losses.append(metrics.get('train_loss', 0))
    
    cb = LambdaCallback(on_epoch_end=log_loss)
    
    # Simulate
    cb.on_epoch_end(None, 1, {'train_loss': 0.5})
    cb.on_epoch_end(None, 2, {'train_loss': 0.3})
    
    print(f"✓ Lambda callback captured losses: {epoch_losses}")


# =============================================================================
# METRICS EXAMPLES
# =============================================================================

def example_19_metric_tracking():
    """Example 19: Track metrics during training."""
    print("\n" + "="*70)
    print("📌 Example 19: Metric Tracking")
    print("="*70)
    
    tracker = MetricTracker()
    
    # Simulate batches
    for _ in range(5):
        output = torch.randn(32, 10)
        target = torch.randint(0, 10, (32,))
        tracker.update(output, target, loss=0.5)
    
    metrics = tracker.compute()
    print(f"  → Accuracy: {metrics['accuracy']:.2f}%")
    print(f"  → Precision: {metrics['precision']:.4f}")
    print(f"  → Recall: {metrics['recall']:.4f}")
    print(f"  → F1: {metrics['f1']:.4f}")
    print("✓ Metric tracking works!")


# =============================================================================
# PROFILING & EXPORT EXAMPLES
# =============================================================================

def example_20_model_profiling():
    """Example 20: Profile model performance."""
    print("\n" + "="*70)
    print("📌 Example 20: Model Profiling")
    print("="*70)
    
    model = CNN("resnet18", num_classes=10)
    profile = model.profile()
    
    print(f"  → Params: {profile['total_params']:,}")
    print(f"  → Size: {profile['model_size_mb']:.2f} MB")
    print(f"  → Latency: {profile['latency_ms']:.2f} ms")
    print("✓ Profiling works!")


def example_21_export_torchscript():
    """Example 21: Export to TorchScript."""
    print("\n" + "="*70)
    print("📌 Example 21: Export to TorchScript")
    print("="*70)
    
    import tempfile
    model = CNN("resnet18", num_classes=10)
    model._build_model()
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        model.export(f.name, format="torchscript")
        print(f"✓ Exported to: {f.name}")


def example_22_quantization():
    """Example 22: Quantize model for faster inference."""
    print("\n" + "="*70)
    print("📌 Example 22: Model Quantization")
    print("="*70)
    
    model = CNN("resnet18", num_classes=10)
    model._build_model()
    model.quantize(bits=8)
    
    # Test inference
    x = torch.randn(1, 3, 224, 224)
    output = model.predict(x)
    print(f"✓ Quantized model output: {output.shape}")


# =============================================================================
# ADVANCED TRAINING EXAMPLES
# =============================================================================

def example_23_gradient_clipping():
    """Example 23: Training with gradient clipping."""
    print("\n" + "="*70)
    print("📌 Example 23: Gradient Clipping")
    print("="*70)
    
    CNN("resnet18", num_classes=10).train(
        "synthetic",
        epochs=1,
        max_grad_norm=1.0,  # Clip gradients
        fast_dev_run=True,
        num_workers=0
    )
    print("✓ Training with gradient clipping works!")


def example_24_label_smoothing():
    """Example 24: Label smoothing regularization."""
    print("\n" + "="*70)
    print("📌 Example 24: Label Smoothing")
    print("="*70)
    
    CNN("resnet18", num_classes=10).train(
        "synthetic",
        epochs=1,
        label_smoothing=0.1,
        fast_dev_run=True,
        num_workers=0
    )
    print("✓ Label smoothing works!")


def example_25_weight_decay():
    """Example 25: Different weight decay values."""
    print("\n" + "="*70)
    print("📌 Example 25: Weight Decay")
    print("="*70)
    
    CNN("resnet18", num_classes=10).train(
        "synthetic",
        epochs=1,
        weight_decay=0.05,  # Stronger regularization
        fast_dev_run=True,
        num_workers=0
    )
    print("✓ Weight decay works!")


# =============================================================================
# CONFIG OBJECT EXAMPLES
# =============================================================================

def example_26_config_object():
    """Example 26: Using config objects."""
    print("\n" + "="*70)
    print("📌 Example 26: Using Config Objects")
    print("="*70)
    
    # Create training config
    config = TrainingConfig(
        epochs=10,
        lr=0.001,
        optimizer="adamw",
        scheduler="cosine",
        batch_size=64,
    )
    print(f"  → Config: epochs={config.epochs}, lr={config.lr}, opt={config.optimizer}")
    print("✓ Config objects work!")


def example_27_model_config():
    """Example 27: Using model-specific configs."""
    print("\n" + "="*70)
    print("📌 Example 27: Model-Specific Configs")
    print("="*70)
    
    cnn_config = CNNConfig(
        num_classes=100,
        pretrained=True,
        drop_rate=0.2,
    )
    
    vit_config = TransformerConfig(
        num_classes=100,
        img_size=224,
        patch_size=16,
    )
    
    print(f"  → CNN Config: {cnn_config.num_classes} classes, drop={cnn_config.drop_rate}")
    print(f"  → ViT Config: img={vit_config.img_size}, patch={vit_config.patch_size}")
    print("✓ Model configs work!")


# =============================================================================
# CUSTOM MODEL EXAMPLES
# =============================================================================

def example_28_custom_model():
    """Example 28: Using a custom PyTorch model."""
    print("\n" + "="*70)
    print("📌 Example 28: Custom PyTorch Model")
    print("="*70)
    
    # Define custom model
    class MyCustomCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Linear(64, num_classes)
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)
    
    # Use with kitepy
    custom_model = MyCustomCNN(num_classes=10)
    model = CNN(custom_model)
    model.train("synthetic", epochs=1, fast_dev_run=True, num_workers=0)
    print("✓ Custom model training works!")


# =============================================================================
# PREDICTION EXAMPLES
# =============================================================================

def example_29_single_prediction():
    """Example 29: Making predictions."""
    print("\n" + "="*70)
    print("📌 Example 29: Making Predictions")
    print("="*70)
    
    model = CNN("resnet18", num_classes=10)
    model._build_model()
    
    # Single image prediction
    image = torch.randn(1, 3, 224, 224)
    output = model.predict(image)
    
    predicted_class = output.argmax(dim=1).item()
    confidence = torch.softmax(output, dim=1).max().item()
    
    print(f"  → Predicted class: {predicted_class}")
    print(f"  → Confidence: {confidence:.2%}")
    print("✓ Prediction works!")


def example_30_batch_prediction():
    """Example 30: Batch predictions."""
    print("\n" + "="*70)
    print("📌 Example 30: Batch Predictions")
    print("="*70)
    
    model = Transformer("vit_tiny", num_classes=10)
    model._build_model()
    
    # Batch of images
    batch = torch.randn(8, 3, 224, 224)
    outputs = model.predict(batch)
    
    predictions = outputs.argmax(dim=1)
    print(f"  → Batch predictions: {predictions.tolist()}")
    print("✓ Batch prediction works!")


# =============================================================================
# UNWRAP EXAMPLES
# =============================================================================

def example_31_unwrap_model():
    """Example 31: Get underlying PyTorch model."""
    print("\n" + "="*70)
    print("📌 Example 31: Unwrap to Raw PyTorch")
    print("="*70)
    
    model = CNN("resnet18", num_classes=10)
    model._build_model()
    
    # Get the raw PyTorch model
    raw_model = model.unwrap()
    
    print(f"  → Type: {type(raw_model).__name__}")
    print(f"  → Is nn.Module: {isinstance(raw_model, nn.Module)}")
    print("✓ Unwrap works!")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print_version_info()
    print_device_info()
    
    print("\n" + "="*70)
    print("🚀 COMPREHENSIVE CNN & ViT EXAMPLES")
    print("="*70)
    
    # BASIC
    example_1_simplest_cnn()
    example_2_simplest_vit()
    example_3_with_config()
    
    # ARCHITECTURES
    example_4_different_cnn_architectures()
    example_5_different_vit_architectures()
    
    # OPTIMIZERS
    example_6_all_optimizers()
    example_7_custom_optimizer()
    
    # SCHEDULERS
    example_8_all_schedulers()
    example_9_warmup_scheduler()
    
    # LOSS FUNCTIONS
    example_10_all_loss_functions()
    example_11_custom_loss()
    
    # AUGMENTATION
    example_12_all_augmentations()
    
    # TRANSFER LEARNING
    example_13_pretrained_model()
    example_14_freeze_backbone()
    example_15_gradual_unfreezing()
    
    # CALLBACKS
    example_16_early_stopping()
    example_17_model_checkpoint()
    example_18_lambda_callback()
    
    # METRICS
    example_19_metric_tracking()
    
    # PROFILING & EXPORT
    example_20_model_profiling()
    example_21_export_torchscript()
    example_22_quantization()
    
    # ADVANCED TRAINING
    example_23_gradient_clipping()
    example_24_label_smoothing()
    example_25_weight_decay()
    
    # CONFIGS
    example_26_config_object()
    example_27_model_config()
    
    # # CUSTOM MODEL
    example_28_custom_model()
    
    # PREDICTIONS
    example_29_single_prediction()
    example_30_batch_prediction()
    
    # UNWRAP
    example_31_unwrap_model()
    
    print("\n" + "="*70)
    print("🎉 ALL 31 EXAMPLES COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("""
Summary of all features demonstrated:
  ✓ 4 Different CNN architectures
  ✓ 4 Different ViT architectures  
  ✓ 7 Different optimizers + custom
  ✓ 6 Different schedulers + warmup
  ✓ 2 Built-in losses + custom loss
  ✓ 6 Augmentation strategies
  ✓ Transfer learning (freeze/unfreeze)
  ✓ 3 Callback types (EarlyStopping, Checkpoint, Lambda)
  ✓ Full metric tracking (accuracy, precision, recall, F1)
  ✓ Model profiling (params, memory, latency)
  ✓ Export (TorchScript)
  ✓ Quantization (INT8)
  ✓ Gradient clipping, label smoothing, weight decay
  ✓ Config objects (TrainingConfig, CNNConfig, TransformerConfig)
  ✓ Custom PyTorch models
  ✓ Single and batch predictions
  ✓ Unwrap to raw PyTorch

This library is PRODUCTION READY! 🚀
""")


if __name__ == "__main__":
    main()
