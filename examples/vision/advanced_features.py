"""
Example: Advanced Features Demo

This demonstrates the powerful advanced features of kitepy.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from kitepy import CNN, Transformer, print_device_info, print_version_info


def main():
    print_version_info()
    print_device_info()
    
    print("\n" + "="*70)
    print("🚀 ADVANCED FEATURES DEMO")
    print("="*70)
    
    # =========================================================================
    # 1. Transfer Learning (Freeze/Unfreeze)
    # =========================================================================
    print("\n" + "-"*70)
    print("1️⃣  TRANSFER LEARNING")
    print("-"*70)
    
    # Load pretrained model and freeze backbone
    model = CNN("resnet50", num_classes=10, pretrained=True)
    model.freeze_backbone()
    
    # Train only the classifier head
    print("\n→ Training with frozen backbone (transfer learning)...")
    model.train("synthetic", epochs=1, fast_dev_run=True, num_workers=0)
    
    # Later, unfreeze for fine-tuning
    model.unfreeze()
    print("\n→ Unfroze all layers for fine-tuning")
    
    # =========================================================================
    # 2. Model Profiling
    # =========================================================================
    print("\n" + "-"*70)
    print("2️⃣  MODEL PROFILING")
    print("-"*70)
    
    model = CNN("efficientnet_b0", num_classes=10)
    profile = model.profile()
    
    print(f"\n📊 Summary: {profile['total_params']:,} params, "
          f"{profile['model_size_mb']:.1f} MB, "
          f"{profile['latency_ms']:.1f} ms latency")
    
    # =========================================================================
    # 3. Model Export (Production Deployment)
    # =========================================================================
    print("\n" + "-"*70)
    print("3️⃣  MODEL EXPORT")
    print("-"*70)
    
    model = CNN("resnet18", num_classes=10)
    model._build_model()
    
    # Export to TorchScript for production
    model.export("exports/model_scripted.pt", format="torchscript")
    
    # Export to ONNX for cross-platform deployment
    try:
        model.export("exports/model.onnx", format="onnx")
    except Exception as e:
        print(f"⚠️  ONNX export skipped: {e}")
    
    # =========================================================================
    # 4. Quantization
    # =========================================================================
    print("\n" + "-"*70)
    print("4️⃣  QUANTIZATION")
    print("-"*70)
    
    model = CNN("resnet18", num_classes=10)
    model._build_model()
    
    # Profile before quantization
    pre_profile = model.profile()
    
    # Quantize to INT8
    model.quantize(bits=8)
    
    # Test inference
    x = torch.randn(1, 3, 224, 224)
    output = model.predict(x)
    print(f"\n✓ Quantized model prediction shape: {output.shape}")
    
    # =========================================================================
    # 5. Advanced Augmentation
    # =========================================================================
    print("\n" + "-"*70)
    print("5️⃣  ADVANCED AUGMENTATION OPTIONS")
    print("-"*70)
    
    print("""
Available augmentation strategies:
  • "none"           - No augmentation
  • "light" / "auto" - Basic (flip, crop)
  • "heavy"          - Color jitter, rotation
  • "randaugment"    - RandAugment (SOTA)
  • "trivialaugment" - TrivialAugment (SOTA, simpler)
  • "autoaugment"    - AutoAugment (ImageNet policy)

Usage:
  from kitepy import get_train_transforms
  transforms = get_train_transforms(224, augmentation="randaugment")
""")
    
    # =========================================================================
    # 6. Vision Transformers
    # =========================================================================
    print("\n" + "-"*70)
    print("6️⃣  VISION TRANSFORMERS (ViT)")
    print("-"*70)
    
    # Create ViT with simplified name
    vit = Transformer("vit_tiny", num_classes=10)
    vit_profile = vit.profile()
    
    print(f"\nViT-Tiny: {vit_profile['total_params']:,} params")
    
    # Freeze and fine-tune
    vit.freeze_backbone()
    vit.train("synthetic", epochs=1, fast_dev_run=True, num_workers=0)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("🎉 ADVANCED FEATURES DEMO COMPLETE!")
    print("="*70)
    print("""
✓ Transfer Learning  - freeze_backbone(), unfreeze()
✓ Model Profiling    - profile() → params, memory, latency
✓ Model Export       - export() → TorchScript, ONNX
✓ Quantization       - quantize() → INT8 for faster inference
✓ Augmentation       - randaugment, trivialaugment, autoaugment
✓ Vision Transformers - Transformer("vit_base") just works
""")


if __name__ == "__main__":
    main()
