import unittest
import sys
import os
import tempfile
import shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kitepy.core.config import (
    TrainingConfig, 
    CNNConfig, 
    DataConfig, 
    ExperimentConfig,
    merge_configs,
    validate_config,
    auto_batch_size,
    auto_learning_rate
)

class TestConfig(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_training_config_defaults(self):
        config = TrainingConfig()
        self.assertIsInstance(config.lr, float)
        self.assertTrue(config.epochs > 0)
        self.assertEqual(config.optimizer, "adamw")

    def test_config_override(self):
        config = TrainingConfig(epochs=50, lr=0.01)
        self.assertEqual(config.epochs, 50)
        self.assertEqual(config.lr, 0.01)
        
        # Test validation logic
        with self.assertRaises(ValueError):
            validate_config(TrainingConfig(epochs=-1, max_steps=None))

    def test_serialization_yaml(self):
        config = TrainingConfig(epochs=100)
        path = os.path.join(self.test_dir, "config.yaml")
        
        config.to_yaml(path)
        self.assertTrue(os.path.exists(path))
        
        loaded = TrainingConfig.from_yaml(path)
        self.assertEqual(loaded.epochs, 100)
        self.assertEqual(loaded.optimizer, config.optimizer)

    def test_serialization_json(self):
        config = CNNConfig(depth=50)
        path = os.path.join(self.test_dir, "config.json")
        
        config.to_json(path)
        self.assertTrue(os.path.exists(path))
        
        loaded = CNNConfig.from_json(path)
        self.assertEqual(loaded.depth, 50)
        self.assertEqual(loaded.arch, "resnet")

    def test_merge_configs(self):
        default = TrainingConfig(epochs=10, lr=1e-3)
        preset = {"epochs": 20, "batch_size": 64}
        user = {"lr": 1e-4}
        
        # Merge: default < preset < user < kwargs
        merged = merge_configs(
            default, 
            preset=preset, 
            user_config=user, 
            epochs=30 # kwargs override
        )
        
        self.assertEqual(merged.epochs, 30)       # kwargs wins
        self.assertEqual(merged.batch_size, 64)   # preset wins over default
        self.assertEqual(merged.lr, 1e-4)         # user wins over default

    def test_auto_tuning(self):
        # Base check
        bs = auto_batch_size("base", available_memory_gb=24.0)
        self.assertEqual(bs, 64)
        
        bs_small_mem = auto_batch_size("base", available_memory_gb=12.0)
        self.assertEqual(bs_small_mem, 32)
        
        lr = auto_learning_rate(batch_size=256, base_lr=1e-3)
        self.assertEqual(lr, 1e-3)
        
        lr_scaled = auto_learning_rate(batch_size=512, base_lr=1e-3)
        self.assertEqual(lr_scaled, 2e-3)

    def test_experiment_config(self):
        exp = ExperimentConfig()
        self.assertIsInstance(exp.model, CNNConfig) # Default
        self.assertIsInstance(exp.training, TrainingConfig)
        
        exp_dict = exp.to_dict()
        self.assertIn("model", exp_dict)
        self.assertIn("training", exp_dict)

if __name__ == '__main__':
    unittest.main()
