"""
Vision model APIs - CNN and Transformer.

These are the public-facing classes users import.
"""

import torch.nn as nn
from typing import Optional, Union, Dict, Any

from kitepy.core.base import BaseModel
from kitepy.core.config import CNNConfig, TransformerConfig
from .wrappers import create_model


# ============================================================================
# CNN API
# ============================================================================

class CNN(BaseModel):
    """
    Convolutional Neural Network for image classification.
    
    Supports ResNet, VGG, EfficientNet, MobileNet, DenseNet, and more.
    
    Examples:
        >>> # Simple usage
        >>> model = CNN("resnet18")
        >>> model.train("cifar10")
        
        >>> # With config
        >>> model = CNN("resnet50", num_classes=100, pretrained=True)
        >>> model.train("cifar100", epochs=50, lr=0.001)
        
        >>> # Custom model
        >>> import torch.nn as nn
        >>> custom_model = nn.Sequential(...)
        >>> model = CNN(custom_model)
        >>> model.train("my_dataset")
    """
    
    modality = "vision"
    default_config_class = CNNConfig
    
    def __init__(
        self,
        model: Optional[Union[str, nn.Module]] = "resnet18",
        config: Optional[Union[Dict, CNNConfig, str]] = None,
        **kwargs
    ):
        """
        Initialize CNN model.
        
        Args:
            model: Model name (e.g., "resnet18") or custom PyTorch model
            config: Configuration dict/object or path to yaml/json
            **kwargs: Override any config values
        """
        super().__init__(model, config, **kwargs)
    
    def _build_model(self):
        """Build the CNN model."""
        if self.custom_model is not None:
            # User provided a custom model
            self.model = self.custom_model
        else:
            # Create from preset/timm
            self.model = create_model(
                model_name_or_module=self.model_name,
                modality=self.modality,
                config=self.config.to_dict(),
                pretrained=self.config.pretrained,
            )
        
        print(f"✓ Created CNN model: {self.model_name}")
        print(f"  - Architecture: {self.config.arch if hasattr(self.config, 'arch') else 'custom'}")
        print(f"  - Parameters: {sum(p.numel() for p in self.model.parameters()):,}")


# ============================================================================
# TRANSFORMER API (Vision Transformers)
# ============================================================================

class Transformer(BaseModel):
    """
    Vision Transformer for image classification.
    
    Supports ViT, DeiT, Swin, and other transformer-based vision models.
    
    Examples:
        >>> # Simple usage
        >>> model = Transformer("vit_base")
        >>> model.train("imagenet")
        
        >>> # With config
        >>> model = Transformer("swin_tiny", img_size=384, pretrained=True)
        >>> model.train("imagenet", epochs=100)
    """
    
    modality = "vision"
    default_config_class = TransformerConfig
    
    def __init__(
        self,
        model: Optional[Union[str, nn.Module]] = "vit_base",
        config: Optional[Union[Dict, TransformerConfig, str]] = None,
        **kwargs
    ):
        """
        Initialize Vision Transformer model.
        
        Args:
            model: Model name (e.g., "vit_base") or custom PyTorch model
            config: Configuration dict/object or path to yaml/json
            **kwargs: Override any config values
        """
        super().__init__(model, config, **kwargs)
    
    def _build_model(self):
        """Build the Vision Transformer model."""
        if self.custom_model is not None:
            self.model = self.custom_model
        else:
            self.model = create_model(
                model_name_or_module=self.model_name,
                modality=self.modality,
                config=self.config.to_dict(),
                pretrained=self.config.pretrained,
            )
        
        print(f"✓ Created Transformer model: {self.model_name}")
        print(f"  - Architecture: {self.config.arch if hasattr(self.config, 'arch') else 'custom'}")
        print(f"  - Parameters: {sum(p.numel() for p in self.model.parameters()):,}")


# ============================================================================
# VISION API (General - for explicit task specification)
# ============================================================================

class Vision(BaseModel):
    """
    General vision model API with explicit task specification.
    
    This is for advanced users who want to specify the exact task.
    
    Examples:
        >>> # Image classification
        >>> model = Vision(task="classification", model="resnet50")
        
        >>> # Object detection (future)
        >>> model = Vision(task="detection", model="yolo")
        
        >>> # Segmentation (future)
        >>> model = Vision(task="segmentation", model="unet")
    """
    
    modality = "vision"
    default_config_class = CNNConfig
    
    def __init__(
        self,
        task: str = "classification",
        model: Optional[Union[str, nn.Module]] = "resnet18",
        config: Optional[Union[Dict, CNNConfig, str]] = None,
        **kwargs
    ):
        """
        Initialize Vision model with task specification.
        
        Args:
            task: Task type ("classification", "detection", "segmentation")
            model: Model name or custom PyTorch model
            config: Configuration dict/object or path to yaml/json
            **kwargs: Override any config values
        """
        self.task = task
        
        if task not in ["classification"]:
            raise NotImplementedError(
                f"Task '{task}' not yet implemented. "
                "Currently supported: classification. "
                "Detection and segmentation coming in Phase 2."
            )
        
        super().__init__(model, config, **kwargs)
    
    def _build_model(self):
        """Build model based on task."""
        if self.task == "classification":
            # Same as CNN
            if self.custom_model is not None:
                self.model = self.custom_model
            else:
                self.model = create_model(
                    model_name_or_module=self.model_name,
                    modality=self.modality,
                    config=self.config.to_dict(),
                    pretrained=self.config.pretrained,
                )
        
        print(f"✓ Created Vision model for {self.task}: {self.model_name}")


# ============================================================================
# EXPORT
# ============================================================================

__all__ = ['CNN', 'Transformer', 'Vision']