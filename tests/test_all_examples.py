"""
Test runner for comprehensive examples.
Runs all examples with exception handling to identify failures.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import traceback
import torch
import torch.nn as nn


def run_with_catch(name, func):
    """Run a function and catch any exceptions."""
    print(f"\n{'='*60}")
    print(f"TESTING: {name}")
    print('='*60)
    try:
        func()
        print(f"[PASS] {name}")
        return True
    except Exception as e:
        print(f"[FAIL] {name}")
        print(f"  Error: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def example_1_simplest_cnn():
    from kitepy import CNN
    CNN("resnet18").train("synthetic", epochs=1, fast_dev_run=True, num_workers=0)


def example_2_simplest_vit():
    from kitepy import Transformer
    Transformer("vit_tiny").train("synthetic", epochs=1, fast_dev_run=True, num_workers=0)


def example_3_with_config():
    from kitepy import CNN
    model = CNN("resnet18", num_classes=10)
    model.train("synthetic", epochs=1, batch_size=32, lr=0.001, fast_dev_run=True, num_workers=0)


def example_4_different_cnn_architectures():
    from kitepy import CNN
    for arch in ["resnet18", "efficientnet_b0"]:
        model = CNN(arch, num_classes=10)
        model._build_model()
        print(f"  {arch}: {sum(p.numel() for p in model.model.parameters()):,} params")


def example_5_different_vit_architectures():
    from kitepy import Transformer
    for arch in ["vit_tiny", "deit_tiny"]:
        model = Transformer(arch, num_classes=10)
        model._build_model()
        print(f"  {arch}: {sum(p.numel() for p in model.model.parameters()):,} params")


def example_6_optimizers():
    from kitepy import CNN
    for opt in ["adam", "sgd", "radam"]:
        CNN("resnet18", num_classes=10).train("synthetic", epochs=1, optimizer=opt, fast_dev_run=True, num_workers=0)
        print(f"  {opt} OK")


def example_7_custom_optimizer():
    from kitepy import CNN
    CNN("resnet18", num_classes=10).train("synthetic", epochs=1, optimizer=torch.optim.Adadelta, fast_dev_run=True, num_workers=0)


def example_8_schedulers():
    from kitepy import CNN
    for sched in ["cosine", "step", "linear"]:
        CNN("resnet18", num_classes=10).train("synthetic", epochs=1, scheduler=sched, fast_dev_run=True, num_workers=0)
        print(f"  {sched} OK")


def example_9_warmup_scheduler():
    from kitepy import CNN
    CNN("resnet18", num_classes=10).train("synthetic", epochs=1, scheduler="warmup_cosine", fast_dev_run=True, num_workers=0)


def example_10_loss_functions():
    from kitepy import CNN
    for loss in ["cross_entropy", "focal"]:
        CNN("resnet18", num_classes=10).train("synthetic", epochs=1, loss=loss, fast_dev_run=True, num_workers=0)
        print(f"  {loss} OK")


def example_11_custom_loss():
    from kitepy import CNN
    
    class MyLoss(nn.Module):
        def forward(self, pred, target):
            return nn.functional.cross_entropy(pred, target)
    
    CNN("resnet18", num_classes=10).train(
        "synthetic", 
        epochs=1, 
        loss=MyLoss(), 
        fast_dev_run=True, 
        num_workers=0,
        save_every_n_epochs=0)


def example_12_augmentations():
    from kitepy import get_train_transforms
    for aug in ["none", "light", "heavy", "randaugment"]:
        transform = get_train_transforms(224, aug)
        print(f"  {aug}: {len(transform.transforms)} transforms")


def example_13_pretrained():
    from kitepy import CNN
    model = CNN("resnet18", num_classes=10, pretrained=True)
    model._build_model()
    print(f"  Pretrained: {sum(p.numel() for p in model.model.parameters()):,} params")


def example_14_freeze_backbone():
    from kitepy import CNN
    model = CNN("resnet50", num_classes=10, pretrained=True)
    model.freeze_backbone()
    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.model.parameters())
    print(f"  Frozen: {trainable:,} / {total:,} trainable")


def example_15_unfreeze():
    from kitepy import CNN
    model = CNN("resnet18", num_classes=10)
    model.freeze_backbone()
    model.unfreeze()
    print("  Unfreeze OK")


def example_16_early_stopping():
    from kitepy import EarlyStopping
    es = EarlyStopping(patience=3, monitor='val_loss')
    for epoch, val_loss in enumerate([0.5, 0.4, 0.45, 0.46, 0.47]):
        es.on_epoch_end(None, epoch, {'val_loss': val_loss})
        if es.should_stop:
            print(f"  Would stop at epoch {epoch}")
            break


def example_17_model_checkpoint():
    from kitepy import ModelCheckpoint
    mc = ModelCheckpoint(filepath="checkpoints/test.pt", save_best_only=True)
    print(f"  ModelCheckpoint: save_best_only={mc.save_best_only}")


def example_18_lambda_callback():
    from kitepy import LambdaCallback
    called = {'count': 0}
    def on_epoch_end(trainer, epoch, metrics, **kwargs):
        called['count'] += 1
    
    cb = LambdaCallback(on_epoch_end=on_epoch_end)
    cb.on_epoch_end(None, 1, {})
    print(f"  Lambda called: {called['count']} times")


def example_19_metrics():
    from kitepy import MetricTracker
    tracker = MetricTracker()
    output = torch.randn(32, 10)
    target = torch.randint(0, 10, (32,))
    tracker.update(output, target, loss=0.5)
    metrics = tracker.compute()
    print(f"  Accuracy: {metrics['accuracy']:.1f}%, F1: {metrics['f1']:.3f}")


def example_20_profiling():
    from kitepy import CNN
    model = CNN("resnet18", num_classes=10)
    profile = model.profile()
    print(f"  Params: {profile['total_params']:,}, Latency: {profile['latency_ms']:.1f}ms")


def example_21_export():
    import tempfile
    from kitepy import CNN
    model = CNN("resnet18", num_classes=10)
    model._build_model()
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        model.export(f.name, format="torchscript")
    print("  Export OK")


def example_22_quantization():
    from kitepy import CNN
    model = CNN("resnet18", num_classes=10)
    model._build_model()
    model.quantize(bits=8)
    x = torch.randn(1, 3, 224, 224)
    output = model.predict(x)
    print(f"  Quantized output: {output.shape}")


def example_23_gradient_clipping():
    from kitepy import CNN
    CNN("resnet18", num_classes=10).train("synthetic", epochs=1, max_grad_norm=1.0, fast_dev_run=True, num_workers=0)


def example_24_label_smoothing():
    from kitepy import CNN
    CNN("resnet18", num_classes=10).train("synthetic", epochs=1, label_smoothing=0.1, fast_dev_run=True, num_workers=0)


def example_25_config_object():
    from kitepy import TrainingConfig
    config = TrainingConfig(epochs=10, lr=0.001, optimizer="adamw")
    print(f"  Config: epochs={config.epochs}, lr={config.lr}")


def example_26_model_config():
    from kitepy import CNNConfig, TransformerConfig
    cnn_config = CNNConfig(num_classes=100)
    vit_config = TransformerConfig(num_classes=100)
    print(f"  CNN: {cnn_config.num_classes} classes, ViT: {vit_config.num_classes} classes")


def example_27_custom_model():
    from kitepy import CNN
    
    class MyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, 10)
        def forward(self, x):
            x = self.pool(self.conv(x))
            return self.fc(x.view(x.size(0), -1))
    
    model = CNN(MyNet())
    model.train("synthetic", epochs=1, fast_dev_run=True, num_workers=0)


def example_28_prediction():
    from kitepy import CNN
    model = CNN("resnet18", num_classes=10)
    model._build_model()
    image = torch.randn(1, 3, 224, 224)
    output = model.predict(image)
    pred = output.argmax(dim=1).item()
    print(f"  Prediction: class {pred}")


def example_29_batch_prediction():
    from kitepy import Transformer
    model = Transformer("vit_tiny", num_classes=10)
    model._build_model()
    batch = torch.randn(4, 3, 224, 224)
    outputs = model.predict(batch)
    preds = outputs.argmax(dim=1).tolist()
    print(f"  Batch predictions: {preds}")


def example_30_unwrap():
    from kitepy import CNN
    model = CNN("resnet18", num_classes=10)
    model._build_model()
    raw = model.unwrap()
    print(f"  Unwrapped type: {type(raw).__name__}")


def main():
    examples = [
        ("1. Simplest CNN", example_1_simplest_cnn),
        ("2. Simplest ViT", example_2_simplest_vit),
        ("3. With Config", example_3_with_config),
        ("4. CNN Architectures", example_4_different_cnn_architectures),
        ("5. ViT Architectures", example_5_different_vit_architectures),
        ("6. Optimizers", example_6_optimizers),
        ("7. Custom Optimizer", example_7_custom_optimizer),
        ("8. Schedulers", example_8_schedulers),
        ("9. Warmup Scheduler", example_9_warmup_scheduler),
        ("10. Loss Functions", example_10_loss_functions),
        ("11. Custom Loss", example_11_custom_loss),
        ("12. Augmentations", example_12_augmentations),
        ("13. Pretrained", example_13_pretrained),
        ("14. Freeze Backbone", example_14_freeze_backbone),
        ("15. Unfreeze", example_15_unfreeze),
        ("16. Early Stopping", example_16_early_stopping),
        ("17. Model Checkpoint", example_17_model_checkpoint),
        ("18. Lambda Callback", example_18_lambda_callback),
        ("19. Metrics", example_19_metrics),
        ("20. Profiling", example_20_profiling),
        ("21. Export", example_21_export),
        ("22. Quantization", example_22_quantization),
        ("23. Gradient Clipping", example_23_gradient_clipping),
        ("24. Label Smoothing", example_24_label_smoothing),
        ("25. Config Object", example_25_config_object),
        ("26. Model Config", example_26_model_config),
        ("27. Custom Model", example_27_custom_model),
        ("28. Prediction", example_28_prediction),
        ("29. Batch Prediction", example_29_batch_prediction),
        ("30. Unwrap", example_30_unwrap),
    ]
    
    passed = 0
    failed = 0
    failures = []
    
    for name, func in examples:
        if run_with_catch(name, func):
            passed += 1
        else:
            failed += 1
            failures.append(name)
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"PASSED: {passed}")
    print(f"FAILED: {failed}")
    
    if failures:
        print("\nFailed examples:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("\nALL EXAMPLES PASSED!")
    
    print("="*60)


if __name__ == "__main__":
    main()
