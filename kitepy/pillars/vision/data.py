"""
Universal data loading engine for all modalities.

Handles vision, text, audio, and multimodal datasets.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Tuple, Union, Any
from pathlib import Path
import warnings

from kitepy.core.config import DataConfig, TrainingConfig
from kitepy.core.presets import DATASET_PRESETS


# ============================================================================
# SYNTHETIC DATA (For testing)
# ============================================================================

class SyntheticDataset(Dataset):
    """Synthetic dataset for testing training loops without downloads."""
    
    def __init__(
        self,
        num_samples: int = 100,
        img_size: int = 224,
        num_classes: int = 10,
        transform=None
    ):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes
        self.transform = transform
        
        # Generate random data once (reproducible if needed)
        self.data = torch.randn(num_samples, 3, img_size, img_size)
        self.targets = torch.randint(0, num_classes, (num_samples,))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]
        
        # Note: In a real dataset, you'd apply transforms to PIL images
        # Here we just simulate it
        if self.transform:
            # We skip actual PIL conversion for speed in synthetic mode
            pass
            
        return sample, target


def _load_custom_folder(
    data_path: Path,
    train_transform,
    val_transform,
    train_config: TrainingConfig,
    val_split: float = 0.2
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Load custom image folder dataset.
    
    Expects folder structure:
    data_path/
      class1/
        img1.jpg
        img2.jpg
      class2/
        img1.jpg
        ...
    
    OR:
    data_path/
      train/
        class1/...
        class2/...
      val/
        class1/...
        class2/...
    """
    from torchvision import datasets
    from torch.utils.data import random_split
    
    train_path = data_path / "train"
    val_path = data_path / "val"
    
    # Check if pre-split exists
    if train_path.is_dir() and val_path.is_dir():
        train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_path, transform=val_transform)
        print(f"✓ Loaded custom dataset from {data_path}")
        print(f"  - Train: {len(train_dataset)} samples, {len(train_dataset.classes)} classes")
        print(f"  - Val: {len(val_dataset)} samples")
    else:
        # Auto-split
        full_dataset = datasets.ImageFolder(data_path, transform=train_transform)
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(train_config.seed)
        )
        
        print(f"✓ Loaded custom dataset from {data_path}")
        print(f"  - Auto-split: {train_size} train, {val_size} val")
        print(f"  - Classes: {len(full_dataset.classes)}")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory,
    )
    
    return train_loader, val_loader, None


# ============================================================================
# VISION DATA LOADING
# ============================================================================

def load_vision_data(
    dataset_name: str,
    data_config: DataConfig,
    train_config: TrainingConfig,
    download: bool = True
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Load vision datasets (CIFAR, ImageNet, etc.).
    
    Args:
        dataset_name: Dataset name or path
        data_config: Data configuration
        train_config: Training configuration
        download: Auto-download if not available
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    try:
        from torchvision import datasets, transforms
    except ImportError:
        raise ImportError(
            "torchvision is required for vision datasets. "
            "Install with: pip install torchvision"
        )
    
    # Check if it's a preset
    if dataset_name in DATASET_PRESETS:
        preset = DATASET_PRESETS[dataset_name]
        img_size = preset.get('img_size', 224)
        num_classes = preset.get('num_classes', 1000)
    else:
        img_size = 224
        num_classes = 10
    
    # Define transforms
    train_transform = get_train_transforms(img_size, data_config.augmentation)
    val_transform = get_val_transforms(img_size)
    
    # Check if it's a custom folder path
    data_path = Path(dataset_name)
    if data_path.is_dir():
        return _load_custom_folder(
            data_path, train_transform, val_transform, train_config
        )
    
    # Load dataset
    if dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(
            root='./data',
            train=True,
            download=download,
            transform=train_transform
        )
        val_dataset = datasets.CIFAR10(
            root='./data',
            train=False,
            download=download,
            transform=val_transform
        )
        test_dataset = None  # CIFAR10 uses same test set
    
    elif dataset_name == "cifar100":
        train_dataset = datasets.CIFAR100(
            root='./data',
            train=True,
            download=download,
            transform=train_transform
        )
        val_dataset = datasets.CIFAR100(
            root='./data',
            train=False,
            download=download,
            transform=val_transform
        )
        test_dataset = None
    
    elif dataset_name == "mnist":
        train_dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=download,
            transform=train_transform
        )
        val_dataset = datasets.MNIST(
            root='./data',
            train=False,
            download=download,
            transform=val_transform
        )
        test_dataset = None
    
    elif dataset_name == "fashion_mnist" or dataset_name == "fashion-mnist":
        train_dataset = datasets.FashionMNIST(
            root='./data',
            train=True,
            download=download,
            transform=train_transform
        )
        val_dataset = datasets.FashionMNIST(
            root='./data',
            train=False,
            download=download,
            transform=val_transform
        )
        test_dataset = None
    
    elif dataset_name == "imagenet" or dataset_name == "imagenet-1k":
        # ImageNet requires manual download
        imagenet_path = Path("./data/imagenet")
        if not imagenet_path.exists():
            raise ValueError(
                f"ImageNet not found at {imagenet_path}. "
                "Please download ImageNet manually and place it in ./data/imagenet"
            )
        
        train_dataset = datasets.ImageFolder(
            root=str(imagenet_path / "train"),
            transform=train_transform
        )
        val_dataset = datasets.ImageFolder(
            root=str(imagenet_path / "val"),
            transform=val_transform
        )
        test_dataset = None
    
    elif Path(dataset_name).exists():
        # Custom dataset from path (assume ImageFolder structure)
        data_path = Path(dataset_name)
        
        train_dataset = datasets.ImageFolder(
            root=str(data_path / "train"),
            transform=train_transform
        )
        
        val_path = data_path / "val"
        if val_path.exists():
            val_dataset = datasets.ImageFolder(
                root=str(val_path),
                transform=val_transform
            )
        else:
            val_dataset = None
        
        test_path = data_path / "test"
        if test_path.exists():
            test_dataset = datasets.ImageFolder(
                root=str(test_path),
                transform=val_transform
            )
        else:
            test_dataset = None
    
    elif dataset_name == "synthetic":
        # Create synthetic data for testing
        num_samples = 128
        train_dataset = SyntheticDataset(
            num_samples=num_samples,
            img_size=img_size,
            num_classes=num_classes,
            transform=train_transform
        )
        val_dataset = SyntheticDataset(
            num_samples=train_config.batch_size * 2,
            img_size=img_size,
            num_classes=num_classes,
            transform=val_transform
        )
        test_dataset = None
    
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASET_PRESETS.keys()) + ['synthetic']}"
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=data_config.shuffle,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory,
        drop_last=data_config.drop_last,
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            num_workers=train_config.num_workers,
            pin_memory=train_config.pin_memory,
        )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            num_workers=train_config.num_workers,
            pin_memory=train_config.pin_memory,
        )
    
    return train_loader, val_loader, test_loader


def get_train_transforms(img_size: int, augmentation: str = "auto"):
    """
    Get training transforms with augmentation.
    
    Args:
        img_size: Target image size
        augmentation: Augmentation level:
            - "none": No augmentation
            - "light" / "auto": Basic augmentation (flip, crop)
            - "heavy": Strong augmentation (color jitter, rotation)
            - "randaugment": RandAugment (state-of-the-art)
            - "trivialaugment": TrivialAugment (SOTA, simpler)
            - "autoaugment": AutoAugment (ImageNet policy)
    """
    from torchvision import transforms
    
    # Check if grayscale (for MNIST, Fashion-MNIST)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    
    if img_size == 28:  # MNIST/Fashion-MNIST
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    
    if augmentation == "none":
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            normalize,
        ])
    
    elif augmentation == "light" or augmentation == "auto":
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    
    elif augmentation == "heavy":
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ])
    
    elif augmentation == "randaugment":
        # RandAugment - state-of-the-art augmentation
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            normalize,
        ])
    
    elif augmentation == "trivialaugment":
        # TrivialAugment - simpler, often better than RandAugment
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            normalize,
        ])
    
    elif augmentation == "autoaugment":
        # AutoAugment with ImageNet policy
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            normalize,
        ])
    
    else:
        raise ValueError(f"Unknown augmentation: {augmentation}. "
                        f"Available: none, light, auto, heavy, randaugment, trivialaugment, autoaugment")


def get_val_transforms(img_size: int):
    """Get validation/test transforms (no augmentation)."""
    from torchvision import transforms
    
    # Check if grayscale (for MNIST, Fashion-MNIST)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    
    if img_size == 28:  # MNIST/Fashion-MNIST
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    
    return transforms.Compose([
        transforms.Resize(img_size + 32),  # Slightly larger
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])


# ============================================================================
# TEXT DATA LOADING
# ============================================================================

def load_text_data(
    dataset_name: str,
    data_config: DataConfig,
    train_config: TrainingConfig,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Load text datasets (WikiText, OpenWebText, etc.).
    
    Args:
        dataset_name: Dataset name
        data_config: Data configuration
        train_config: Training configuration
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets (HuggingFace) is required for text datasets. "
            "Install with: pip install datasets"
        )
    
    # Load from HuggingFace datasets
    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-103-v1")
    elif dataset_name == "openwebtext":
        dataset = load_dataset("openwebtext")
    elif dataset_name == "c4":
        dataset = load_dataset("c4", "en", streaming=True)
    else:
        # Try loading directly
        dataset = load_dataset(dataset_name)
    
    # TODO: Tokenization and collation
    # This will be fully implemented in Phase 3 (LLM)
    raise NotImplementedError(
        "Text data loading will be fully implemented in Phase 3 (LLM module)"
    )


# ============================================================================
# MULTIMODAL DATA LOADING
# ============================================================================

def load_multimodal_data(
    dataset_name: str,
    data_config: DataConfig,
    train_config: TrainingConfig,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Load multimodal datasets (COCO, Flickr30k, etc.).
    
    Args:
        dataset_name: Dataset name
        data_config: Data configuration
        train_config: Training configuration
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    # TODO: Implement in Phase 4 (VLM)
    raise NotImplementedError(
        "Multimodal data loading will be implemented in Phase 4 (VLM module)"
    )


# ============================================================================
# UNIVERSAL DATA LOADER
# ============================================================================

def load_data(
    dataset_name: str,
    modality: str,
    data_config: DataConfig,
    train_config: TrainingConfig,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Universal data loader - routes to appropriate loader based on modality.
    
    Args:
        dataset_name: Dataset name or path
        modality: "vision", "language", "multimodal"
        data_config: Data configuration
        train_config: Training configuration
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    if modality == "vision":
        return load_vision_data(dataset_name, data_config, train_config)
    
    elif modality == "language":
        return load_text_data(dataset_name, data_config, train_config)
    
    elif modality == "multimodal":
        return load_multimodal_data(dataset_name, data_config, train_config)
    
    else:
        raise ValueError(f"Unknown modality: {modality}")


# ============================================================================
# CUSTOM DATASET WRAPPER
# ============================================================================

class CustomDatasetWrapper(Dataset):
    """Wrapper for custom user datasets."""
    
    def __init__(self, dataset: Dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        if self.transform:
            item = self.transform(item)
        
        return item


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    'load_data',
    'load_vision_data',
    'load_text_data',
    'load_multimodal_data',
    'get_train_transforms',
    'get_val_transforms',
    'CustomDatasetWrapper',
]