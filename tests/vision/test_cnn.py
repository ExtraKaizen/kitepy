"""
Test suite for CNN models.

Run with: pytest tests/test_cnn.py -v
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kitepy import CNN
from kitepy import CNNConfig, TrainingConfig


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Create temporary checkpoint directory."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return str(checkpoint_dir)


# ============================================================================
# BASIC TESTS
# ============================================================================

def test_cnn_creation():
    """Test CNN model creation."""
    model = CNN("resnet18")
    assert model is not None
    assert model.model_name == "resnet18"
    assert model.modality == "vision"


def test_cnn_with_config():
    """Test CNN with custom config."""
    config = CNNConfig(
        num_classes=10,
        dropout=0.5,
        pretrained=False
    )
    model = CNN("resnet18", config=config)
    assert model.config.num_classes == 10
    assert model.config.dropout == 0.5


def test_cnn_with_kwargs():
    """Test CNN with kwargs override."""
    model = CNN("resnet18", num_classes=100, dropout=0.3)
    assert model.config.num_classes == 100
    assert model.config.dropout == 0.3


def test_custom_pytorch_model():
    """Test CNN with custom PyTorch model."""
    custom_model = nn.Sequential(
        nn.Conv2d(3, 64, 3),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 10)
    )
    
    model = CNN(custom_model)
    assert model.custom_model is not None
    assert model.model_name == "custom"


# ============================================================================
# MODEL BUILDING TESTS
# ============================================================================

def test_model_build():
    """Test that model builds correctly."""
    model = CNN("resnet18")
    model._build_model()
    
    assert model.model is not None
    assert isinstance(model.model, nn.Module)


def test_model_forward_pass():
    """Test forward pass through model."""
    model = CNN("resnet18", num_classes=10)
    model._build_model()
    
    # Create dummy input
    x = torch.randn(2, 3, 224, 224)
    output = model.model(x)
    
    assert output.shape == (2, 10)


# ============================================================================
# TRAINING TESTS (Quick tests - 1 epoch)
# ============================================================================

# @pytest.mark.slow
# def test_cnn_training_cifar10(temp_checkpoint_dir):
#     """Test CNN training on CIFAR-10 (1 epoch)."""
#     model = CNN("resnet18", num_classes=10)
    
#     # Train for 1 epoch only (for testing)
#     model.train(
#         data="cifar10",
#         epochs=1,
#         batch_size=64,
#         checkpoint_dir=temp_checkpoint_dir,
#         save_every_n_epochs=1,
#     )
    
#     assert model.is_trained
#     assert len(model.training_history) == 1
#     assert 'train_loss' in model.training_history[0]


@pytest.mark.slow
def test_cnn_fast_dev_run():
    """Test fast dev run (1 batch for debugging)."""
    model = CNN("resnet18", num_classes=10)
    
    # This should run very quickly
    model.train(
        data="cifar10",
        epochs=1,
        batch_size=64,
        fast_dev_run=True,
    )
    
    assert model.is_trained


# ============================================================================
# INFERENCE TESTS
# ============================================================================

def test_cnn_predict():
    """Test prediction."""
    model = CNN("resnet18", num_classes=10)
    model._build_model()
    model.device = torch.device("cpu")
    model.model = model.model.to(model.device)
    
    # Create dummy input
    x = torch.randn(1, 3, 224, 224)
    output = model.predict(x)
    
    assert output.shape == (1, 10)


# ============================================================================
# CHECKPOINT TESTS
# ============================================================================

def test_save_and_load(temp_checkpoint_dir):
    """Test saving and loading checkpoints."""
    # Create and save model
    model1 = CNN("resnet18", num_classes=10)
    model1._build_model()
    
    checkpoint_path = f"{temp_checkpoint_dir}/test_model.pt"
    model1.save(checkpoint_path)
    
    assert Path(checkpoint_path).exists()
    
    # Load model
    model2 = CNN.load(checkpoint_path)
    
    assert model2.model_name == "resnet18"
    assert model2.config.num_classes == 10


# ============================================================================
# UTILITY TESTS
# ============================================================================

def test_describe():
    """Test describe method."""
    model = CNN("resnet18")
    # This should not raise an error
    model.describe()


def test_summary():
    """Test summary method."""
    model = CNN("resnet18")
    model._build_model()
    # This should not raise an error
    model.summary()


def test_explain_config():
    """Test explain_config method."""
    model = CNN("resnet18")
    # This should not raise an error
    model.explain_config()


def test_unwrap():
    """Test unwrap to get PyTorch model."""
    model = CNN("resnet18")
    model._build_model()
    
    pytorch_model = model.unwrap()
    assert isinstance(pytorch_model, nn.Module)


# ============================================================================
# PRESET TESTS
# ============================================================================

def test_list_models():
    """Test listing available models."""
    from kitepy import list_models
    
    models = list_models("cnn")
    assert isinstance(models, list)
    assert len(models) > 0
    assert "resnet18" in models


def test_different_presets():
    """Test creating models with different presets."""
    presets = ["resnet18", "resnet50", "efficientnet_b0"]
    
    for preset in presets:
        try:
            model = CNN(preset)
            assert model.model_name == preset
        except Exception as e:
            pytest.skip(f"Preset {preset} not available: {e}")


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

def test_invalid_model_name():
    """Test that invalid model names raise errors."""
    with pytest.raises(ValueError):
        model = CNN("invalid_model_that_does_not_exist")
        model._build_model()


def test_train_without_data():
    """Test that training without data raises error."""
    model = CNN("resnet18")
    
    with pytest.raises(Exception):
        model.train(data=None, epochs=1)


# ============================================================================
# INTEGRATION TEST
# ============================================================================

# @pytest.mark.slow
# @pytest.mark.integration
# def test_full_pipeline(temp_checkpoint_dir):
#     """Test complete pipeline: create -> train -> save -> load -> predict."""
#     # 1. Create model
#     model1 = CNN("resnet18", num_classes=10)
    
#     # 2. Train (1 epoch)
#     model1.train(
#         data="cifar10",
#         epochs=1,
#         batch_size=128,
#         checkpoint_dir=temp_checkpoint_dir,
#     )
    
#     # 3. Save
#     checkpoint_path = f"{temp_checkpoint_dir}/final_model.pt"
#     model1.save(checkpoint_path)
    
#     # 4. Load
#     model2 = CNN.load(checkpoint_path)
    
#     # 5. Predict
#     x = torch.randn(1, 3, 32, 32)  # CIFAR-10 size
#     output = model2.predict(x)
    
#     assert output.shape == (1, 10)
#     print("\n✓ Full pipeline test passed!")


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])