import unittest
import sys
import torch
import torch.nn as nn
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kitepy.core.utils import (
    set_seed,
    get_device,
    count_parameters,
    get_model_size_mb,
    format_time,
    get_checkpoint_path,
    cleanup_old_checkpoints,
    resolve_auto_value,
    Logger
)

class TestUtils(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_set_seed(self):
        set_seed(42)
        a = torch.randn(5)
        
        set_seed(42)
        b = torch.randn(5)
        
        self.assertTrue(torch.allclose(a, b))
        
    def test_get_device(self):
        # Basic check that it returns a device object
        # We can't easily test cuda/mps availability without hardware,
        # but we can test the fallback or explicit str
        device = get_device("cpu")
        self.assertEqual(device.type, "cpu")
        
        # Test error
        with self.assertRaises(ValueError):
            get_device(999) # invalid gpu index

    def test_model_summary_utils(self):
        model = nn.Linear(10, 2) # weights: 2x10=20, bias: 2 -> 22 params
        
        count = count_parameters(model)
        self.assertEqual(count, 22)
        
        # Freeze parameters
        for p in model.parameters():
            p.requires_grad = False
            
        trainable = count_parameters(model, trainable_only=True)
        self.assertEqual(trainable, 0)
        
        size_mb = get_model_size_mb(model)
        self.assertGreater(size_mb, 0)

    def test_format_time(self):
        self.assertEqual(format_time(30), "30.0s")
        self.assertEqual(format_time(90), "1.5m")
        self.assertEqual(format_time(3600), "1.0h")
        self.assertEqual(format_time(3660), "1.0h")

    def test_get_checkpoint_path(self):
        path = get_checkpoint_path(
            self.test_dir, "model", 5, "loss", 0.1234
        )
        self.assertIn("model_epoch5_loss0.1234.pt", path)
        self.assertTrue(Path(self.test_dir).exists())

    def test_cleanup_old_checkpoints(self):
        # Create dummy checkpoints
        import time
        
        # Checkpoint 1: loss 0.5 (worst)
        ckpt1 = Path(self.test_dir) / "model_loss0.5000.pt"
        ckpt1.touch()
        time.sleep(0.01) # ensure modify time separation if needed (though fn uses name)
        
        # Checkpoint 2: loss 0.3
        ckpt2 = Path(self.test_dir) / "model_loss0.3000.pt"
        ckpt2.touch()
        
        # Checkpoint 3: loss 0.1 (best)
        ckpt3 = Path(self.test_dir) / "model_loss0.1000.pt"
        ckpt3.touch()
        
        # Keep only top 1 (best loss)
        cleanup_old_checkpoints(self.test_dir, keep_top_k=1, metric_name="loss", lower_is_better=True)
        
        self.assertTrue(ckpt3.exists(), "Best checkpoint should remain")
        self.assertFalse(ckpt1.exists(), "Worst checkpoint should be deleted")
        self.assertFalse(ckpt2.exists(), "2nd worst checkpoint should be deleted")

        # Keep top 2
        ckpt1.touch() # recreate
        ckpt2.touch()
        cleanup_old_checkpoints(self.test_dir, keep_top_k=2, metric_name="loss", lower_is_better=True)
        self.assertTrue(ckpt3.exists())
        self.assertTrue(ckpt2.exists())
        self.assertFalse(ckpt1.exists())

    def test_resolve_auto_value(self):
        self.assertEqual(resolve_auto_value(10, lambda: 20), 10)
        self.assertEqual(resolve_auto_value("auto", lambda: 20), 20)
        self.assertEqual(resolve_auto_value("AUTO", lambda: 20), 20)
        self.assertEqual(resolve_auto_value("other", lambda: 20), "other")

    @patch('builtins.print')
    def test_logger(self, mock_print):
        log_file = Path(self.test_dir) / "train.log"
        logger = Logger(str(log_file))
        
        logger.info("Test message")
        
        # Check stdout
        mock_print.assert_called_with("[INFO] Test message")
        
        # Check file
        with open(log_file, 'r') as f:
            content = f.read()
        self.assertIn("[INFO] Test message", content)

if __name__ == '__main__':
    unittest.main()
