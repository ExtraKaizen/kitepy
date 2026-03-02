# kitepy - Dead Simple Deep Learning 🚀

Build and train deep learning models with **one line of code**.

```python
from kitepy import CNN
CNN("resnet18").train("cifar10")
```

That's it. No boilerplate. No configuration hell. Just magic. ✨

---

## Features

- **🎯 One-Line Training**: `CNN("resnet18").train("cifar10")`
- **🔧 Zero Configuration**: Smart defaults that just work
- **🎨 Full Customization**: Override anything when needed
- **🌐 Multi-Modal**: CNNs, Transformers, LLMs, VLMs (coming soon)
- **⚡ Production Ready**: Built on PyTorch, timm, HuggingFace
- **🚄 Fast**: Automatic mixed precision, multi-GPU, distributed training

---

## Installation

```bash
# Minimal install (CNN support)
pip install kitepy

# With LLM support
pip install kitepy[llm]

# Full install (all features)
pip install kitepy[all]

# Development install
git clone https://github.com/ExtraKaizen/kitepy
cd kitepy
pip install -e .
```

---

## Quick Start

### Example 1: Simplest Possible

```python
from kitepy import CNN

# Train ResNet-18 on CIFAR-10
CNN("resnet18").train("cifar10")
```

### Example 2: Custom Settings

```python
# Override any parameter
CNN("resnet50").train(
    data="cifar100",
    epochs=50,
    batch_size=128,
    lr=0.001,
)
```

### Example 3: Save and Load

```python
# Train and save
model = CNN("resnet18")
model.train("cifar10", epochs=10)
model.save("my_model.pt")

# Load and use
loaded = CNN.load("my_model.pt")
predictions = loaded.predict(my_images)
```

### Example 4: Custom PyTorch Model

```python
import torch.nn as nn

# Use your own model
my_model = nn.Sequential(
    nn.Conv2d(3, 64, 3),
    nn.ReLU(),
    # ... your architecture
)

CNN(my_model).train("my_dataset")
```

---

## Supported Models

### CNNs (Vision)
- **ResNet**: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`
- **EfficientNet**: `efficientnet_b0` through `efficientnet_b7`
- **MobileNet**: `mobilenet_v2`, `mobilenet_v3_small`, `mobilenet_v3_large`
- **VGG**: `vgg11`, `vgg13`, `vgg16`, `vgg19`
- **DenseNet**: `densenet121`, `densenet169`, `densenet201`
- **And 800+ more from timm!**

### Vision Transformers
- **ViT**: `vit_tiny`, `vit_small`, `vit_base`, `vit_large`, `vit_huge`
- **DeiT**: `deit_tiny`, `deit_small`, `deit_base`
- **Swin**: `swin_tiny`, `swin_small`, `swin_base`, `swin_large`

### Coming Soon (Phase 2+)
- **LLMs**: GPT, LLaMA, Mistral (Phase 3)
- **VLMs**: CLIP, BLIP, Flamingo (Phase 4)
- **Audio**: Whisper, AST (Phase 5)
- **Diffusion**: Stable Diffusion (Phase 5)

---

## Supported Datasets

### Built-in Vision Datasets
- `cifar10`, `cifar100`
- `mnist`, `fashion_mnist`
- `imagenet` (requires manual download)

### Custom Datasets
```python
# From path (ImageFolder structure)
CNN("resnet18").train("/path/to/my/dataset")

# From PyTorch DataLoader
from torch.utils.data import DataLoader
my_loader = DataLoader(...)
CNN("resnet18").train(my_loader)
```

---

## Advanced Usage

### Configuration System

```python
from kitepy import CNN, CNNConfig, TrainingConfig

# Method 1: Config objects
model_config = CNNConfig(
    depth=50,
    num_classes=100,
    dropout=0.5
)

train_config = TrainingConfig(
    epochs=100,
    batch_size=256,
    lr=0.01,
    optimizer="adamw",
    scheduler="cosine",
)

model = CNN("resnet50", config=model_config)
model.train("cifar100", **train_config.to_dict())
```

### YAML Configuration

```yaml
# config.yaml
model:
  arch: resnet
  depth: 50
  num_classes: 100

training:
  epochs: 100
  batch_size: 256
  lr: 0.01
  optimizer: adamw
```

```python
model = CNN(config="config.yaml")
model.train("cifar100")
```

### Model Inspection

```python
model = CNN("resnet18")

# Show architecture and config
model.describe()

# Show parameter count
model.summary()

# Show all config options
model.explain_config()

# Get underlying PyTorch model
pytorch_model = model.unwrap()
```

### List Available Models

```python
from kitepy import list_models, list_datasets

# List all CNN models
models = list_models("cnn")
print(models)  # ['resnet18', 'resnet50', ...]

# List all datasets
datasets = list_datasets()
print(datasets)  # ['cifar10', 'imagenet', ...]
```

### Register Custom Presets

```python
from kitepy import register_preset

register_preset("cnn", "my_custom_resnet", {
    "arch": "resnet",
    "depth": 101,
    "width_multiplier": 2.0,
})

# Now use it
CNN("my_custom_resnet").train("imagenet")
```

---

## API Reference

### CNN

```python
CNN(
    model: str | nn.Module,           # Model name or custom PyTorch model
    config: dict | Config | str,      # Config dict, object, or YAML path
    **kwargs                          # Override any config value
)
```

**Methods:**
- `.train(data, epochs, **kwargs)` - Train the model
- `.evaluate(data)` - Evaluate on test data
- `.predict(input)` - Run inference
- `.save(path)` - Save checkpoint
- `.load(path)` - Load checkpoint (classmethod)
- `.describe()` - Print model info
- `.summary()` - Print parameter count
- `.explain_config()` - Show config options
- `.unwrap()` - Get PyTorch model

---

## Architecture

```
kitepy/
├── api/           # Public APIs (CNN, LLM, VLM, etc.)
├── core/          # Core engines (training, data, optimization)
├── config/        # Configuration system
├── models/        # Architecture templates
├── data/          # Data loading
├── wrappers/      # timm, HuggingFace wrappers
└── utils/         # Utilities
```

**Key Principle**: We wrap existing ecosystems (PyTorch, timm, HuggingFace) instead of reimplementing.

---

## Development Roadmap

- [x] **Phase 1**: CNN Module ✅
- [x] **Phase 2**: Vision Transformers ✅
- [ ] **Phase 3**: LLM Module (Coming Soon)
- [ ] **Phase 4**: VLM Module (Planned)
- [ ] **Phase 5**: Audio & Diffusion (Planned)

---

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- timm >= 0.9.0 (for vision models)
- transformers >= 4.30.0 (for LLMs, optional)

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](https://github.com/ExtraKaizen/kitepy/blob/main/CONTRIBUTING.md) for guidelines.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Citation

If you use kitepy in your research, please cite:

```bibtex
@software{kitepy2026,
  title={kitepy: Dead Simple Deep Learning},
  author={ExtraKaizen},
  year={2026},
  url={https://github.com/ExtraKaizen/kitepy}
}
```

---

## Acknowledgments

Built on the shoulders of giants:
- [PyTorch](https://pytorch.org/)
- [timm](https://github.com/huggingface/pytorch-image-models)
- [HuggingFace Transformers](https://huggingface.co/transformers)
- [PyTorch Lightning](https://lightning.ai/)

---

**Made with ❤️ for the AI community**