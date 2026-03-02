"""
Quickstart example for Vision Transformer (ViT) training.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from kitepy import Transformer, print_device_info, print_version_info


def main():
    print_version_info()
    print_device_info()
    
    # Check if we have a GPU
    has_gpu = torch.cuda.is_available()
    
    print("\n" + "="*70)
    print("Example 1: ViT Training Demo")
    print("="*70)
    
    if not has_gpu:
        print("⚠️  No GPU detected - using fast_dev_run=True for quick demo")
        print("   (Full ViT training on CPU would take many hours)")
        print()
    
    # Note: ViT models expect 224x224 images by default
    # CIFAR-10 is 32x32, so we use synthetic 224x224 data for this demo
    print("ℹ️  Using synthetic 224x224 data (ViT requires 224x224 input)")
    print()
    
    # Train ViT-Tiny (smallest, fastest)
    model = Transformer("vit_tiny", num_classes=10)
    
    if has_gpu:
        model.train("synthetic", epochs=2, batch_size=32)
    else:
        model.train("synthetic", epochs=1, fast_dev_run=True, num_workers=0)
    
    print("\n✓ Example 1 complete!")
    
    
    print("\n" + "="*70)
    print("Example 2: Different ViT Sizes")
    print("="*70)
    
    models = ["vit_tiny", "vit_small", "vit_base"]
    
    for model_name in models:
        print(f"\n→ Testing {model_name}...")
        model = Transformer(model_name, num_classes=10)
        model._build_model()
        
        num_params = sum(p.numel() for p in model.model.parameters())
        print(f"✓ {model_name}: {num_params:,} parameters")
    
    
    print("\n" + "="*70)
    print("Example 3: Fast Dev Run (Smoke Test)")
    print("="*70)
    
    model = Transformer("vit_tiny", num_classes=10)
    model.train("synthetic", epochs=1, fast_dev_run=True, num_workers=0)
    
    print("\n✓ All ViT examples complete!")


if __name__ == "__main__":
    main()