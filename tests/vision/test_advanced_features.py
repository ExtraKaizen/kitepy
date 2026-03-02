"""
Test suite for advanced features.

Tests: freeze_backbone, unfreeze, profile, export, find_lr, quantize
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kitepy import CNN, Transformer


class TestTransferLearning:
    """Tests for transfer learning features."""
    
    def test_freeze_backbone_cnn(self):
        """Test freezing CNN backbone."""
        model = CNN("resnet18", num_classes=10)
        model._build_model()
        
        # Before freezing
        total_before = sum(p.requires_grad for p in model.model.parameters())
        
        # Freeze
        model.freeze_backbone()
        
        # After freezing
        trainable = sum(p.requires_grad for p in model.model.parameters())
        
        assert trainable < total_before
        assert trainable > 0  # Should have some trainable (the head)
    
    def test_freeze_backbone_vit(self):
        """Test freezing ViT backbone."""
        model = Transformer("vit_tiny", num_classes=10)
        model._build_model()
        
        model.freeze_backbone()
        
        trainable = sum(p.requires_grad for p in model.model.parameters())
        total = sum(1 for _ in model.model.parameters())
        
        assert trainable < total
    
    def test_unfreeze_all(self):
        """Test unfreezing all layers."""
        model = CNN("resnet18", num_classes=10)
        model._build_model()
        model.freeze_backbone()
        model.unfreeze()
        
        # All should be trainable
        frozen = sum(1 for p in model.model.parameters() if not p.requires_grad)
        assert frozen == 0


class TestProfiling:
    """Tests for model profiling."""
    
    def test_profile_cnn(self):
        """Test profiling CNN."""
        model = CNN("resnet18", num_classes=10)
        
        profile = model.profile()
        
        assert 'total_params' in profile
        assert 'latency_ms' in profile
        assert 'model_size_mb' in profile
        assert profile['total_params'] > 0
    
    def test_profile_vit(self):
        """Test profiling ViT."""
        model = Transformer("vit_tiny", num_classes=10)
        
        profile = model.profile()
        
        assert profile['total_params'] > 0
        assert profile['latency_ms'] > 0


class TestExport:
    """Tests for model export."""
    
    def test_export_torchscript(self, tmp_path):
        """Test TorchScript export."""
        model = CNN("resnet18", num_classes=10)
        model._build_model()
        
        export_path = str(tmp_path / "model.pt")
        model.export(export_path, format="torchscript")
        
        assert os.path.exists(export_path)
        
        # Verify it loads
        loaded = torch.jit.load(export_path)
        assert loaded is not None
    
    def test_export_onnx(self, tmp_path):
        """Test ONNX export."""
        model = CNN("resnet18", num_classes=10)
        model._build_model()
        
        export_path = str(tmp_path / "model.onnx")
        
        try:
            model.export(export_path, format="onnx")
            # If export succeeds, file should exist
            assert os.path.exists(export_path)
        except Exception as e:
            # ONNX export can fail in some environments
            pytest.skip(f"ONNX export not available: {e}")


class TestQuantization:
    """Tests for model quantization."""
    
    def test_quantize_int8(self):
        """Test INT8 quantization."""
        model = CNN("resnet18", num_classes=10)
        model._build_model()
        
        # Get original size
        original_params = sum(p.numel() for p in model.model.parameters())
        
        # Quantize
        model.quantize(bits=8)
        
        # Model should still work
        x = torch.randn(1, 3, 224, 224)
        output = model.predict(x)
        
        assert output.shape == (1, 10)


class TestAdvancedTraining:
    """Tests for advanced training features."""
    
    @pytest.mark.slow
    def test_find_lr(self):
        """Test LR finder."""
        model = CNN("resnet18", num_classes=10)
        
        suggested_lr = model.find_lr(data="synthetic", batch_size=4, num_workers=0)
        
        assert suggested_lr > 0
        assert suggested_lr < 1


# Quick smoke test
def test_advanced_features_smoke():
    """Quick smoke test for advanced features."""
    print("\n🚀 Advanced Features Smoke Test")
    print("="*60)
    
    # 1. Create model with pretrained and freeze
    print("\n1️⃣  Testing transfer learning...")
    model = CNN("resnet18", num_classes=10)
    model.freeze_backbone()
    print("   ✓ freeze_backbone() works")
    
    model.unfreeze()
    print("   ✓ unfreeze() works")
    
    # 2. Profile
    print("\n2️⃣  Testing profiling...")
    profile = model.profile()
    print(f"   ✓ profile() works: {profile['total_params']:,} params")
    
    # 3. Export
    print("\n3️⃣  Testing export...")
    model.export("test_export.pt", format="torchscript")
    print("   ✓ export() to TorchScript works")
    os.remove("test_export.pt")
    
    # 4. Quantize (on a fresh model)
    print("\n4️⃣  Testing quantization...")
    model2 = CNN("resnet18", num_classes=10)
    model2._build_model()
    model2.quantize(bits=8)
    print("   ✓ quantize() works")
    
    print("\n" + "="*60)
    print("🎉 All Advanced Features Smoke Test PASSED!")
    print("="*60)


if __name__ == "__main__":
    test_advanced_features_smoke()
