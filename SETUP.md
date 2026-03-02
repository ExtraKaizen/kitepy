# Setup Instructions 🛠️

Follow these steps to install and test the CNN module.

---

## Step 1: Install Dependencies

```bash
# Install core dependencies
pip install torch torchvision timm pyyaml numpy

# Optional: For testing
pip install pytest
```

---

## Step 2: Install kitepy (Development Mode)

```bash
# Clone the repository
git clone https://github.com/ExtraKaizen/kitepy
cd kitepy

# Install in development mode
pip install -e .
```

This installs kitepy so you can import it from anywhere.

---

## Step 3: Verify Installation

```bash
# Open Python and test import
python -c "from kitepy import CNN; print('✓ kitepy installed successfully!')"
```

---

## Step 4: Run Quick Test

Create a file `test_quick.py`:

```python
from kitepy import CNN

print("Testing CNN creation...")
model = CNN("resnet18", num_classes=10)
print("✓ CNN created successfully!")

print("\nModel info:")
model.describe()
```

Run it:
```bash
python test_quick.py
```

---

## Step 5: Run Full Training Test (Optional)

**Note**: This will download CIFAR-10 (~170MB) and train for 2 epochs (~5 minutes on GPU).

```python
# test_training.py
from kitepy import CNN

print("Training CNN on CIFAR-10...")
model = CNN("resnet18", num_classes=10)
model.train("cifar10", epochs=2, batch_size=64)
print("✓ Training completed!")
```

Run it:
```bash
python test_training.py
```

---

## Step 6: Run Pytest Suite

```bash
# Run all tests
pytest tests/test_cnn.py -v

# Run only fast tests (skip training tests)
pytest tests/test_cnn.py -v -m "not slow"

# Run specific test
pytest tests/test_cnn.py::test_cnn_creation -v
```

---

## Step 7: Run Example Scripts

```bash
# Run quickstart example (includes training)
python examples/quickstart_cnn.py
```

**Warning**: This will train multiple models and may take 10-30 minutes.

---

## Common Issues & Solutions

### Issue 1: "No module named 'kitepy'"

**Solution**: Make sure you installed in development mode:
```bash
pip install -e .
```

### Issue 2: "CUDA out of memory"

**Solution**: Reduce batch size:
```python
CNN("resnet18").train("cifar10", batch_size=32)  # Instead of 64
```

### Issue 3: "Dataset CIFAR10 not found"

**Solution**: The dataset will auto-download on first use. Make sure you have internet connection and ~200MB free space.

### Issue 4: timm model not found

**Solution**: Update timm:
```bash
pip install --upgrade timm
```

---

## Minimal Working Example

If you just want to see it work ASAP:

```python
# minimal_test.py
from kitepy import CNN
import torch

# Create model
model = CNN("resnet18", num_classes=10)
model._build_model()

# Test forward pass
x = torch.randn(2, 3, 224, 224)
output = model.model(x)

print(f"✓ Input shape: {x.shape}")
print(f"✓ Output shape: {output.shape}")
print("✓ Model works!")
```

---

## Project Structure Check

Make sure your directory looks like this:

```
kitepy/
├── kitepy/
│   ├── core/           # Core infrastructure
│   ├── pillars/        # Domain-specific modules (Vision)
│   └── __init__.py     # Public API exports
├── tests/              # Test suite
├── examples/           # Usage examples
├── docs/               # Mintlify documentation
├── pyproject.toml      # Package configuration
├── README.md           # Main documentation
└── SETUP.md            # This file
```

---

## Next Steps

Once everything works:

1. ✅ CNN module is ready!
2. ⏭️ Next: Vision Transformers (Coming Soon)
3. ⏭️ Planned: LLM and VLM support

---

## Getting Help

If you encounter issues:

1. Check this SETUP.md for common solutions
2. Check the error message carefully
3. Make sure all dependencies are installed
4. Try the minimal working example above
5. Check if your PyTorch installation is working:
   ```python
   import torch
   print(torch.__version__)
   print(torch.cuda.is_available())
   ```

---

## Development Workflow

For active development:

```bash
# 1. Make changes to code
# 2. No need to reinstall (using -e mode)
# 3. Test changes
python test_quick.py

# 4. Run tests
pytest tests/test_cnn.py -v

# 5. Commit changes
git add .
git commit -m "Added feature X"
```

---

**Happy coding! 🚀**