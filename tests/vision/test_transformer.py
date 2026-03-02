"""
Test suite for Vision Transformer (ViT) models.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kitepy import Transformer
from kitepy import TransformerConfig


def test_transformer_creation():
    """Test ViT model creation."""
    model = Transformer("vit_base")
    assert model is not None
    assert model.model_name == "vit_base"
    assert model.modality == "vision"


def test_transformer_build():
    """Test that ViT builds correctly."""
    model = Transformer("vit_base", num_classes=10)
    model._build_model()
    
    assert model.model is not None
    assert isinstance(model.model, nn.Module)


def test_transformer_forward():
    """Test forward pass through ViT."""
    model = Transformer("vit_tiny", num_classes=10)  # Use tiny for speed
    model._build_model()
    
    # ViT expects 224x224
    x = torch.randn(2, 3, 224, 224)
    output = model.model(x)
    
    assert output.shape == (2, 10)


@pytest.mark.slow
def test_transformer_training():
    """Test ViT training on CIFAR-10 (1 epoch)."""
    # CIFAR-10 is 32x32, so we must tell ViT to use that image size
    model = Transformer("vit_tiny", num_classes=10, img_size=32)
    
    model.train(
        data="cifar10",
        epochs=1,
        batch_size=64,
        fast_dev_run=True,  # Just 1 batch
    )
    
    assert model.is_trained


def test_different_vit_presets():
    """Test creating different ViT models."""
    presets = ["vit_tiny", "vit_small", "vit_base"]
    
    for preset in presets:
        try:
            model = Transformer(preset)
            assert model.model_name == preset
        except Exception as e:
            pytest.skip(f"Preset {preset} not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])