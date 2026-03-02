import unittest
import sys
import os
import tempfile
import shutil
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kitepy import (
    EarlyStopping,
    ModelCheckpoint,
    CallbackList,
    LambdaCallback,
    TrainingConfig
)

# Dummy Trainer for mocking
class DummyTrainer:
    def __init__(self):
        self.model = nn.Linear(1, 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.config = TrainingConfig()

class TestCallbacks(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.trainer = DummyTrainer()
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_early_stopping(self):
        es = EarlyStopping(monitor='val_loss', patience=2, mode='min')
        
        # Improve
        es.on_epoch_end(None, 1, {'val_loss': 10.0})
        self.assertEqual(es.best_value, 10.0)
        self.assertEqual(es.counter, 0)
        
        # Improve again
        es.on_epoch_end(None, 2, {'val_loss': 9.0})
        self.assertEqual(es.best_value, 9.0)
        
        # No improve
        es.on_epoch_end(None, 3, {'val_loss': 9.5})
        self.assertEqual(es.counter, 1)
        
        # No improve again -> Trigger
        es.on_epoch_end(None, 4, {'val_loss': 9.5})
        self.assertEqual(es.counter, 2)
        self.assertTrue(es.should_stop)

    def test_model_checkpoint(self):
        # Save every time improved
        ckpt_path = os.path.join(self.test_dir, "model_{epoch}.pt")
        mc = ModelCheckpoint(
            filepath=ckpt_path,
            monitor='val_loss',
            mode='min',
            save_best_only=True
        )
        
        # Epoch 1: 10.0 (Best)
        mc.on_epoch_end(self.trainer, 1, {'val_loss': 10.0})
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "model_1.pt")))
        
        # Epoch 2: 11.0 (Worse) -> No save
        mc.on_epoch_end(self.trainer, 2, {'val_loss': 11.0})
        self.assertFalse(os.path.exists(os.path.join(self.test_dir, "model_2.pt")))
        
        # Epoch 3: 9.0 (Best) -> Save
        mc.on_epoch_end(self.trainer, 3, {'val_loss': 9.0})
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "model_3.pt")))

    def test_callback_list(self):
        # Mock callbacks
        cb1 = MagicMock()
        cb2 = MagicMock()
        
        cbl = CallbackList([cb1, cb2])
        
        cbl.on_epoch_end(self.trainer, 1, {})
        
        cb1.on_epoch_end.assert_called_once()
        cb2.on_epoch_end.assert_called_once()

    def test_lambda_callback(self):
        mock_fn = MagicMock()
        
        lc = LambdaCallback(on_epoch_end=mock_fn)
        lc.on_epoch_end(self.trainer, 1, {})
        
        mock_fn.assert_called_once_with(self.trainer, 1, {})

if __name__ == '__main__':
    unittest.main()
