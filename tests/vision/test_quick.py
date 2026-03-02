"""Quick test of all comprehensive features."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kitepy import CNN, Transformer, MetricTracker, EarlyStopping, ModelCheckpoint
import torch

print("=== QUICK COMPREHENSIVE TEST ===")

print("\n1. Basic CNN")
CNN("resnet18", num_classes=10).train("synthetic", epochs=1, fast_dev_run=True, num_workers=0)

print("\n2. Basic ViT")
Transformer("vit_tiny", num_classes=10).train("synthetic", epochs=1, fast_dev_run=True, num_workers=0)

print("\n3. Callbacks")
es = EarlyStopping(patience=3)
mc = ModelCheckpoint()
print("   Callbacks OK")

print("\n4. Metrics")
tracker = MetricTracker()
tracker.update(torch.randn(32, 10), torch.randint(0, 10, (32,)), loss=0.5)
metrics = tracker.compute()
print(f"   Accuracy: {metrics['accuracy']:.1f}%")

print("\n5. Profiling")
model = CNN("resnet18", num_classes=10)
profile = model.profile()
print(f"   Latency: {profile['latency_ms']:.1f}ms")

print("\n6. Custom optimizer (Adadelta)")
CNN("resnet18", num_classes=10).train("synthetic", epochs=1, optimizer=torch.optim.Adadelta, fast_dev_run=True, num_workers=0)

print("\n7. Different scheduler")
CNN("resnet18", num_classes=10).train("synthetic", epochs=1, scheduler="step", fast_dev_run=True, num_workers=0)

print("\n8. Focal loss")
CNN("resnet18", num_classes=10).train("synthetic", epochs=1, loss="focal", fast_dev_run=True, num_workers=0)

print("\n9. Transfer learning")
model = CNN("resnet50", num_classes=10, pretrained=True)
model.freeze_backbone()
trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.model.parameters())
print(f"   Frozen: {trainable:,} / {total:,} trainable")

print("\n10. Model export")
import tempfile
model = CNN("resnet18", num_classes=10)
model._build_model()
with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
    model.export(f.name, format="torchscript")
print("   Export OK")

print("\n=== ALL 10 TESTS PASSED ===")
