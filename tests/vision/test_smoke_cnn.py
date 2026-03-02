import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kitepy import CNN
import torch

def test_smoke():
    print("🚀 Running Smoke Test (Fast Dev Run + Synthetic Data)")
    
    # 1. Create model
    # Use resnet18 with 10 classes
    model = CNN("resnet18", num_classes=10)
    
    # 2. Train with fast_dev_run=True and synthetic data
    # This should take seconds and download NOTHING
    model.train(
        data="synthetic",
        epochs=1,
        batch_size=4,
        fast_dev_run=True,
        mixed_precision=False,
        num_workers=0
    )
    
    print("\n✅ Smoke test passed! Training loop is functional.")
    
    # 3. Test prediction
    x = torch.randn(1, 3, 224, 224)
    output = model.predict(x)
    print(f"✅ Prediction works! Output shape: {output.shape}")

if __name__ == "__main__":
    test_smoke()
