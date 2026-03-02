"""
Wrappers for external model libraries (timm, HuggingFace, etc.).

This is where we leverage existing ecosystems instead of reimplementing.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union
import warnings


# ============================================================================
# TIMM NAME MAPPING (User-friendly → Actual timm name)
# ============================================================================

TIMM_NAME_MAPPING = {
    # ViT (Vision Transformer) - Our simplified names → timm names
    "vit_tiny": "vit_tiny_patch16_224",
    "vit_small": "vit_small_patch16_224",
    "vit_base": "vit_base_patch16_224",
    "vit_large": "vit_large_patch16_224",
    "vit_huge": "vit_huge_patch14_224",
    
    # DeiT (Data-efficient Image Transformer)
    "deit_tiny": "deit_tiny_patch16_224",
    "deit_small": "deit_small_patch16_224",
    "deit_base": "deit_base_patch16_224",
    
    # Swin Transformer
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "swin_small": "swin_small_patch4_window7_224",
    "swin_base": "swin_base_patch4_window7_224",
    "swin_large": "swin_large_patch4_window7_224",
    
    # EfficientNet V2 simplified
    "efficientnetv2_s": "tf_efficientnetv2_s",
    "efficientnetv2_m": "tf_efficientnetv2_m",
    "efficientnetv2_l": "tf_efficientnetv2_l",
    
    # ConvNeXt
    "convnext_tiny": "convnext_tiny",
    "convnext_small": "convnext_small",
    "convnext_base": "convnext_base",
    "convnext_large": "convnext_large",
}


def resolve_model_name(user_name: str) -> str:
    """
    Resolve user-friendly name to actual timm model name.
    
    Args:
        user_name: User-provided model name (simplified or exact timm name)
    
    Returns:
        Resolved timm model name
    """
    # Check our mapping first
    if user_name in TIMM_NAME_MAPPING:
        resolved = TIMM_NAME_MAPPING[user_name]
        print(f"ℹ️  Resolved '{user_name}' → '{resolved}'")
        return resolved
    
    # Otherwise, pass through (might be exact timm name or CNN name)
    return user_name


# ============================================================================
# TIMM WRAPPER (Vision Models)
# ============================================================================

def create_timm_model(
    model_name: str,
    num_classes: int = 1000,
    pretrained: bool = False,
    in_channels: int = 3,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    **kwargs
) -> nn.Module:
    """
    Create a vision model from timm.
    
    Args:
        model_name: Model name - can be simplified (e.g., "vit_base") or exact timm name
        num_classes: Number of output classes
        pretrained: Load pretrained weights
        in_channels: Number of input channels
        drop_rate: Dropout rate
        drop_path_rate: Drop path rate (stochastic depth)
        **kwargs: Additional timm model arguments
    
    Returns:
        PyTorch model from timm
    """
    try:
        import timm
    except ImportError:
        raise ImportError(
            "timm is required for vision models. Install with: pip install timm"
        )
    
    # Resolve user-friendly name to actual timm name
    resolved_name = resolve_model_name(model_name)
    
    # Check if model exists in timm
    available_models = timm.list_models()
    if resolved_name not in available_models:
        # Try to find similar models
        similar = [m for m in available_models if model_name.split('_')[0] in m][:5]
        if similar:
            raise ValueError(
                f"Model '{model_name}' (resolved to '{resolved_name}') not found in timm. "
                f"Did you mean one of: {similar}?"
            )
        raise ValueError(f"Model '{resolved_name}' not available in timm")
    
    # Create model
    model = timm.create_model(
        resolved_name,
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=in_channels,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        **kwargs
    )
    
    return model


def list_timm_models(filter: Optional[str] = None) -> list:
    """
    List available timm models.
    
    Args:
        filter: Optional filter string (e.g., "resnet", "vit")
    
    Returns:
        List of model names
    """
    try:
        import timm
    except ImportError:
        return []
    
    models = timm.list_models()
    
    if filter:
        models = [m for m in models if filter.lower() in m.lower()]
    
    return models


def get_timm_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a timm model.
    
    Args:
        model_name: timm model name
    
    Returns:
        Dictionary with model info
    """
    try:
        import timm
    except ImportError:
        return {}
    
    # Create model to extract info
    model = timm.create_model(model_name, pretrained=False)
    
    # Get model config
    config = model.default_cfg
    
    info = {
        'model_name': model_name,
        'input_size': config.get('input_size', (3, 224, 224)),
        'num_classes': config.get('num_classes', 1000),
        'architecture': config.get('architecture', 'unknown'),
        'pretrained_available': config.get('has_weights', False),
    }
    
    return info


# ============================================================================
# HUGGINGFACE WRAPPER (Language & Multimodal Models)
# ============================================================================

def create_hf_model(
    model_name: str,
    model_type: str = "causal_lm",
    num_labels: Optional[int] = None,
    pretrained: bool = False,
    trust_remote_code: bool = False,
    **kwargs
) -> nn.Module:
    """
    Create a model from HuggingFace Transformers.
    
    Args:
        model_name: HF model name or path (e.g., "gpt2", "bert-base-uncased")
        model_type: Type of model:
            - "causal_lm": Autoregressive language model (GPT-style)
            - "masked_lm": Masked language model (BERT-style)
            - "seq2seq": Sequence-to-sequence (T5-style)
            - "classification": Sequence classification
            - "vision": Vision model (ViT, etc.)
        num_labels: Number of labels (for classification)
        pretrained: Load pretrained weights
        trust_remote_code: Trust remote code (for custom models)
        **kwargs: Additional HF model arguments
    
    Returns:
        PyTorch model from HuggingFace
    """
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForMaskedLM,
            AutoModelForSeq2SeqLM,
            AutoModelForSequenceClassification,
            AutoModel,
        )
    except ImportError:
        raise ImportError(
            "transformers is required. Install with: pip install transformers"
        )
    
    # Select appropriate AutoModel class
    model_classes = {
        'causal_lm': AutoModelForCausalLM,
        'masked_lm': AutoModelForMaskedLM,
        'seq2seq': AutoModelForSeq2SeqLM,
        'classification': AutoModelForSequenceClassification,
        'base': AutoModel,
    }
    
    ModelClass = model_classes.get(model_type)
    if ModelClass is None:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Load model
    if pretrained:
        model = ModelClass.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            num_labels=num_labels,
            **kwargs
        )
    else:
        # Load config and initialize from scratch
        from transformers import AutoConfig
        
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            num_labels=num_labels,
            **kwargs
        )
        model = ModelClass.from_config(config)
    
    return model


def create_hf_tokenizer(
    model_name: str,
    trust_remote_code: bool = False,
    **kwargs
):
    """
    Create a tokenizer from HuggingFace.
    
    Args:
        model_name: HF model/tokenizer name
        trust_remote_code: Trust remote code
        **kwargs: Additional tokenizer arguments
    
    Returns:
        HuggingFace tokenizer
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError(
            "transformers is required. Install with: pip install transformers"
        )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        **kwargs
    )
    
    return tokenizer


def list_hf_models(task: Optional[str] = None, limit: int = 100) -> list:
    """
    List popular HuggingFace models.
    
    Args:
        task: Optional task filter (e.g., "text-generation", "image-classification")
        limit: Maximum number of models to return
    
    Returns:
        List of model names
    """
    try:
        from huggingface_hub import list_models
    except ImportError:
        # Return some common models if HF Hub not available
        return [
            "gpt2", "gpt2-medium", "gpt2-large",
            "bert-base-uncased", "bert-large-uncased",
            "t5-small", "t5-base", "t5-large",
        ]
    
    models = list_models(
        task=task,
        sort="downloads",
        direction=-1,
        limit=limit
    )
    
    return [m.modelId for m in models]


# ============================================================================
# CUSTOM MODEL WRAPPER
# ============================================================================

class CustomModelWrapper(nn.Module):
    """
    Wrapper for custom PyTorch models to ensure compatibility.
    
    This ensures user-provided models work with our training engine.
    """
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def __getattr__(self, name):
        """Forward attribute access to wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def wrap_custom_model(model: nn.Module) -> nn.Module:
    """
    Wrap a custom user-provided model.
    
    Args:
        model: User's PyTorch model
    
    Returns:
        Wrapped model
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected nn.Module, got {type(model)}")
    
    return CustomModelWrapper(model)


# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_model(
    model_name_or_module: Union[str, nn.Module],
    modality: str,
    config: Dict[str, Any],
    pretrained: bool = False,
) -> nn.Module:
    """
    Universal model factory - creates models from any source.
    
    Args:
        model_name_or_module: Model name (preset/timm/HF) or custom nn.Module
        modality: "vision", "language", "multimodal"
        config: Model configuration dict
        pretrained: Load pretrained weights
    
    Returns:
        PyTorch model
    """
    # If already a PyTorch module, wrap and return
    if isinstance(model_name_or_module, nn.Module):
        return wrap_custom_model(model_name_or_module)
    
    # Otherwise, create from name
    model_name = model_name_or_module
    
    if modality == "vision":
        # Try timm first (most vision models)
        try:
            model = create_timm_model(
                model_name=model_name,
                num_classes=config.get('num_classes', 1000),
                pretrained=pretrained,
                in_channels=config.get('in_channels', 3),
                drop_rate=config.get('drop_rate', 0.0),
                drop_path_rate=config.get('drop_path_rate', 0.0),
                img_size=config.get('img_size'),
                patch_size=config.get('patch_size'),
            )
            return model
        except ImportError:
            # timm not installed, try HuggingFace
            try:
                model = create_hf_model(
                    model_name=model_name,
                    model_type='base',
                    pretrained=pretrained,
                )
                return model
            except:
                raise ImportError(
                    "Neither timm nor transformers are installed. "
                    "Install with: pip install timm"
                )
        except ValueError as e:
            # Model not found in timm - re-raise with clear message
            raise ValueError(str(e))
    
    elif modality == "language":
        # Use HuggingFace for language models
        model = create_hf_model(
            model_name=model_name,
            model_type=config.get('model_type', 'causal_lm'),
            pretrained=pretrained,
            num_labels=config.get('num_labels'),
        )
        return model
    
    elif modality == "multimodal":
        # For VLMs, we'll need to create combined architectures
        # This will be implemented in Phase 4
        raise NotImplementedError(
            "Multimodal models coming in Phase 4. "
            "For now, use vision and language models separately."
        )
    
    else:
        raise ValueError(f"Unknown modality: {modality}")


# ============================================================================
# MODEL INSPECTION
# ============================================================================

def get_model_source(model_name: str) -> str:
    """
    Determine which library provides a model.
    
    Args:
        model_name: Model name
    
    Returns:
        Source library ("timm", "huggingface", "unknown")
    """
    # Check timm
    try:
        import timm
        if model_name in timm.list_models():
            return "timm"
    except ImportError:
        pass
    
    # Check HuggingFace (harder to check, so just try)
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained(model_name)
        return "huggingface"
    except:
        pass
    
    return "unknown"


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    # timm
    'create_timm_model',
    'list_timm_models',
    'get_timm_model_info',
    
    # HuggingFace
    'create_hf_model',
    'create_hf_tokenizer',
    'list_hf_models',
    
    # Custom
    'wrap_custom_model',
    
    # Factory
    'create_model',
    'get_model_source',
]