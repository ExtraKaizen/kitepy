"""
Test all new comprehensive features.

Tests: callbacks, metrics, custom loss, optimizers, schedulers, augmentations
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from kitepy import (
    CNN, Transformer,
    EarlyStopping, ModelCheckpoint, LRMonitor, LambdaCallback,
    MetricTracker, TrainingConfig
)


def test_callbacks():
    """Test callback system."""
    print("\n1️⃣  Testing Callback System...")
    
    # Test EarlyStopping
    es = EarlyStopping(patience=3)
    assert es.patience == 3
    print("   ✓ EarlyStopping created")
    
    # Test ModelCheckpoint
    mc = ModelCheckpoint(save_best_only=True)
    assert mc.save_best_only == True
    print("   ✓ ModelCheckpoint created")
    
    # Test LambdaCallback
    called = {'count': 0}
    def on_epoch_end(trainer, epoch, metrics, **kwargs):
        called['count'] += 1
    
    lc = LambdaCallback(on_epoch_end=on_epoch_end)
    lc.on_epoch_end(None, 1, {})
    assert called['count'] == 1
    print("   ✓ LambdaCallback works")


def test_metrics():
    """Test metric tracking."""
    print("\n2️⃣  Testing Metrics...")
    
    tracker = MetricTracker()
    
    # Simulate some predictions
    output = torch.randn(32, 10)  # 32 samples, 10 classes
    target = torch.randint(0, 10, (32,))
    
    tracker.update(output, target, loss=0.5)
    metrics = tracker.compute()
    
    assert 'accuracy' in metrics
    assert 'loss' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    
    print(f"   ✓ Metrics computed: acc={metrics['accuracy']:.1f}%, f1={metrics['f1']:.3f}")


def test_training_config():
    """Test new training config fields."""
    print("\n3️⃣  Testing Config Fields...")
    
    config = TrainingConfig(
        loss="focal",
        optimizer="radam",
        scheduler="warmup_cosine",
        mixup_alpha=0.2,
        cutmix_alpha=0.2,
    )
    
    assert config.loss == "focal"
    assert config.optimizer == "radam"
    assert config.scheduler == "warmup_cosine"
    assert config.mixup_alpha == 0.2
    print("   ✓ All new config fields work")


def test_optimizers():
    """Test different optimizers."""
    print("\n4️⃣  Testing Optimizers...")
    
    optimizers = ["adam", "adamw", "sgd", "rmsprop", "radam", "nadam"]
    
    for opt_name in optimizers:
        model = CNN("resnet18", num_classes=10)
        model.train(
            "synthetic", 
            epochs=1, 
            optimizer=opt_name,
            fast_dev_run=True, 
            num_workers=0
        )
        print(f"   ✓ {opt_name} works")


def test_schedulers():
    """Test different schedulers."""
    print("\n5️⃣  Testing Schedulers...")
    
    schedulers = ["cosine", "step", "polynomial", "onecycle"]
    
    for sched_name in schedulers:
        model = CNN("resnet18", num_classes=10)
        model.train(
            "synthetic", 
            epochs=1, 
            scheduler=sched_name,
            fast_dev_run=True, 
            num_workers=0
        )
        print(f"   ✓ {sched_name} works")


def test_loss_functions():
    """Test different loss functions."""
    print("\n6️⃣  Testing Loss Functions...")
    
    losses = ["cross_entropy", "focal"]
    
    for loss_name in losses:
        model = CNN("resnet18", num_classes=10)
        model.train(
            "synthetic", 
            epochs=1, 
            loss=loss_name,
            fast_dev_run=True, 
            num_workers=0
        )
        print(f"   ✓ {loss_name} works")


def main():
    print("\n" + "="*60)
    print("🚀 COMPREHENSIVE FEATURES TEST")
    print("="*60)
    
    test_callbacks()
    test_metrics()
    test_training_config()
    test_optimizers()
    test_schedulers()
    test_loss_functions()
    
    print("\n" + "="*60)
    print("🎉 ALL COMPREHENSIVE FEATURES TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    main()
