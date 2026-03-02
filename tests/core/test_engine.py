import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import shutil
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kitepy import Engine, TrainingConfig

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    def forward(self, x):
        return self.fc(x)

class TestEngine(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
        # Dummy data
        self.X = torch.randn(20, 10)
        self.y = torch.randint(0, 2, (20,))
        self.dataset = TensorDataset(self.X, self.y)
        self.loader = DataLoader(self.dataset, batch_size=4, shuffle=True)
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_engine_fast_dev_run(self):
        config = TrainingConfig(
            fast_dev_run=True,
            checkpoint_dir=self.test_dir,
            devices="cpu", # Force CPU for CI/test env
            mixed_precision=False
        )
        model = SimpleModel()
        engine = Engine(model, config)
        
        # Should run 1 batch and return
        engine.train(self.loader)
        
        self.assertEqual(engine.global_step, 1)

    def test_engine_full_loop(self):
        config = TrainingConfig(
            epochs=2,
            checkpoint_dir=self.test_dir,
            devices="cpu",
            mixed_precision=False,
            save_every_n_epochs=1,
            log_every_n_steps=1 # Ensure logging happens so we don't divide by zero if batch count is small (though default is 50, batch count 20/4=5. 5 % 50 != 0. )
        )
        model = SimpleModel()
        engine = Engine(model, config)
        
        engine.train(self.loader, self.loader) # Train and Val on same data
        
        self.assertEqual(engine.current_epoch, 2)
        self.assertEqual(len(engine.history), 2)
        self.assertTrue(Path(self.test_dir).exists())

if __name__ == '__main__':
    unittest.main()
