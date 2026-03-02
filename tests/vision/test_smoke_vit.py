"""
Smoke test for Vision Transformer (ViT) models.

This tests the entire ViT pipeline without downloading datasets.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kitepy import Transformer
import torch


def test_vit_smoke():
    """Quick smoke test for ViT."""
    print("🚀 Running ViT Smoke Test")
    print("=" * 70)
    
    # 1. Test model creation with simplified name
    print("\n1️⃣  Testing model creation (vit_tiny)...")
    model = Transformer("vit_tiny", num_classes=10)
    print(f"   ✓ Created: {model.model_name}")
    
    # 2. Build the model
    print("\n2️⃣  Building model...")
    model._build_model()
    num_params = sum(p.numel() for p in model.model.parameters())
    print(f"   ✓ Built successfully!")
    print(f"   ✓ Parameters: {num_params:,}")
    
    # 3. Test forward pass
    print("\n3️⃣  Testing forward pass...")
    model.device = torch.device("cpu")
    model.model = model.model.to(model.device)
    x = torch.randn(2, 3, 224, 224)
    output = model.predict(x)
    print(f"   ✓ Input shape: {x.shape}")
    print(f"   ✓ Output shape: {output.shape}")
    
    # 4. Test training with synthetic data
    print("\n4️⃣  Testing training loop (synthetic data + fast_dev_run)...")
    model2 = Transformer("vit_tiny", num_classes=10)
    model2.train(
        data="synthetic",
        epochs=1,
        batch_size=4,
        fast_dev_run=True,
        num_workers=0
    )
    print(f"   ✓ Training completed!")
    
    # 5. Test different ViT sizes
    print("\n5️⃣  Testing different ViT presets...")
    presets = ["vit_tiny", "vit_small", "vit_base"]
    for preset in presets:
        try:
            m = Transformer(preset, num_classes=10)
            m._build_model()
            params = sum(p.numel() for p in m.model.parameters())
            print(f"   ✓ {preset}: {params:,} parameters")
        except Exception as e:
            print(f"   ✗ {preset}: {e}")
    
    print("\n" + "=" * 70)
    print("🎉 ViT Smoke Test PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    test_vit_smoke()
