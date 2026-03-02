"""
Quickstart example for CNN training.

This demonstrates the simplest possible usage of kitepy.
"""

import sys
from pathlib import Path

# Add parent directory to path (for development)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kitepy import CNN, print_device_info, print_version_info


def main():
    """Run CNN training examples."""
    import torch
    
    # Print environment info
    print_version_info()
    print_device_info()
    
    # Check if we have a GPU
    has_gpu = torch.cuda.is_available()
    
    print("\n" + "="*70)
    print("Example 1: Simplest Possible Usage")
    print("="*70)
    
    if not has_gpu:
        print("⚠️  No GPU detected - using fast_dev_run=True for quick demo")
        print("   (Full training on CPU would take hours)")
        print()
    
    # This is ALL you need!
    model = CNN("resnet18")
    
    if has_gpu:
        # Full training on GPU
        model.train("cifar10", epochs=2)
    else:
        # Quick demo on CPU (just 1 batch to verify everything works)
        model.train("cifar10", epochs=1, fast_dev_run=True, num_workers=0)
    
    print("\n✓ Example 1 complete!")
    
    
    # print("\n" + "="*70)
    # print("Example 2: With Custom Settings")
    # print("="*70)
    
    # model = CNN("resnet18", num_classes=10)
    # model.train(
    #     data="cifar10",
    #     epochs=5,
    #     batch_size=128,
    #     lr=0.001,
    # )
    
    # print("\n✓ Example 2 complete!")
    
    
    # print("\n" + "="*70)
    # print("Example 3: Save and Load")
    # print("="*70)
    
    # # Train a model
    # model = CNN("resnet18", num_classes=10)
    # model.train("cifar10", epochs=2)
    
    # # Save it
    # model.save("checkpoints/my_resnet18.pt")
    
    # # Load it back
    # loaded_model = CNN.load("checkpoints/my_resnet18.pt")
    
    # # Use it for inference
    # import torch
    # dummy_input = torch.randn(1, 3, 32, 32)  # CIFAR-10 size
    # output = loaded_model.predict(dummy_input)
    # print(f"\nPrediction output shape: {output.shape}")
    
    # print("\n✓ Example 3 complete!")
    
    
    # print("\n" + "="*70)
    # print("Example 4: Model Inspection")
    # print("="*70)
    
    # model = CNN("resnet50")
    
    # # Describe the model
    # model.describe()
    
    # # Show available config options
    # model.explain_config()
    
    # print("\n✓ Example 4 complete!")
    
    
    # print("\n" + "="*70)
    # print("Example 5: Different Architectures")
    # print("="*70)
    
    # # Try different models
    # models_to_try = ["resnet18", "resnet34", "efficientnet_b0"]
    
    # for model_name in models_to_try:
    #     try:
    #         print(f"\n→ Testing {model_name}...")
    #         model = CNN(model_name, num_classes=10)
    #         model._build_model()
    #         print(f"✓ {model_name} created successfully")
            
    #         # Print parameter count
    #         num_params = sum(p.numel() for p in model.model.parameters())
    #         print(f"  Parameters: {num_params:,}")
        
    #     except Exception as e:
    #         print(f"✗ {model_name} failed: {e}")
    
    # print("\n✓ Example 5 complete!")
    
    
    # print("\n" + "="*70)
    # print("Example 6: Custom PyTorch Model")
    # print("="*70)
    
    # import torch.nn as nn
    
    # # Define a custom model
    # custom_model = nn.Sequential(
    #     nn.Conv2d(3, 32, kernel_size=3, padding=1),
    #     nn.ReLU(),
    #     nn.MaxPool2d(2),
    #     nn.Conv2d(32, 64, kernel_size=3, padding=1),
    #     nn.ReLU(),
    #     nn.MaxPool2d(2),
    #     nn.AdaptiveAvgPool2d(1),
    #     nn.Flatten(),
    #     nn.Linear(64, 10)
    # )
    
    # # Use it with kitepy
    # model = CNN(custom_model)
    # model.train("cifar10", epochs=2)
    
    # print("\n✓ Example 6 complete!")
    
    
    # print("\n" + "="*70)
    # print("🎉 All examples completed successfully!")
    # print("="*70)


if __name__ == "__main__":
    main()