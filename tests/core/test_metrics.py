import unittest
import sys
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kitepy.core.metrics import (
    MetricTracker, 
    accuracy, 
    precision_recall_f1
)

class TestMetrics(unittest.TestCase):
    def test_accuracy_function(self):
        output = torch.tensor([
            [0.1, 0.9, 0.0], # Class 1
            [0.8, 0.1, 0.1], # Class 0
            [0.2, 0.3, 0.5]  # Class 2
        ])
        target = torch.tensor([1, 0, 2])
        
        acc1, = accuracy(output, target, topk=(1,))
        self.assertEqual(acc1, 100.0)
        
        # Test top-k with mismatch
        # target: [0, 0, 0]
        # pred:   [1, 0, 2] -> 1/3 correct
        output_wrong = torch.tensor([
            [0.1, 0.9, 0.0], # Pred 1 (Incorrect, logic wants 0)
            [0.8, 0.1, 0.1], # Pred 0 (Correct)
            [0.2, 0.3, 0.5]  # Pred 2 (Incorrect, logic wants 0)
        ])
        target_wrong = torch.tensor([0, 0, 0])
        acc1, acc2 = accuracy(output_wrong, target_wrong, topk=(1, 2))
        
        self.assertAlmostEqual(acc1, 33.3333, places=2)
        # Top-2: 
        # 1: [0.9(1), 0.1(0)] -> 0 is in top-2? Yes.
        # 2: [0.8(0), 0.1(1)] -> 0 is in top-2? Yes.
        # 3: [0.5(2), 0.3(1)] -> 0 is NOT in top-2.
        # So 2/3 correct
        self.assertAlmostEqual(acc2, 66.6666, places=2)

    def test_precision_recall_f1_function(self):
        # 3 classes
        # Class 0: 1 TP
        # Class 1: 1 TP
        # Class 2: 1 TP
        output = torch.tensor([
            [10, 0, 0], 
            [0, 10, 0], 
            [0, 0, 10]
        ])
        target = torch.tensor([0, 1, 2])
        
        metrics = precision_recall_f1(output, target, average='macro')
        self.assertEqual(metrics['precision'], 1.0)
        self.assertEqual(metrics['recall'], 1.0)
        self.assertEqual(metrics['f1'], 1.0)
        
        # Mixed case
        # Preds: [0, 0, 1]
        # Targets: [0, 1, 2]
        output_mixed = torch.tensor([
            [10, 0, 0],
            [10, 0, 0],
            [0, 10, 0]
        ])
        target_mixed = torch.tensor([0, 1, 2])
        
        # Class 0: TP=1, FP=1, FN=0 -> P=0.5, R=1.0, F1=0.66
        # Class 1: TP=0, FP=1, FN=1 -> P=0.0, R=0.0, F1=0.0
        # Class 2: TP=0, FP=0, FN=1 -> P=0.0, R=0.0, F1=0.0
        
        # Macro: P=(0.5+0+0)/3 = 0.166, R=(1+0+0)/3=0.333
        metrics = precision_recall_f1(output_mixed, target_mixed, average='macro')
        self.assertAlmostEqual(metrics['precision'], 0.1666, places=2)
        self.assertAlmostEqual(metrics['recall'], 0.3333, places=2)

    def test_metric_tracker_reset_update_compute(self):
        tracker = MetricTracker(num_classes=2)
        
        # Batch 1: Perfect
        tracker.update(
            torch.tensor([[10., 0.], [0., 10.]]), 
            torch.tensor([0, 1]),
            loss=1.0
        )
        # Batch 2: Perfect
        tracker.update(
            torch.tensor([[10., 0.], [0., 10.]]), 
            torch.tensor([0, 1]),
            loss=0.5
        )
        
        metrics = tracker.compute()
        self.assertEqual(metrics['accuracy'], 100.0)
        self.assertEqual(metrics['loss'], 0.75) # Avg of 1.0 and 0.5
        
        # Check reset
        tracker.reset()
        self.assertEqual(tracker.total, 0)
        
        # Check empty compute
        metrics_empty = tracker.compute()
        self.assertEqual(metrics_empty['accuracy'], 0.0)

    def test_metric_tracker_confusion_matrix(self):
        tracker = MetricTracker(num_classes=3)
        # Preds: [0, 1, 2]
        # Target: [0, 2, 1]
        output = torch.tensor([
            [10, 0, 0],
            [0, 10, 0],
            [0, 0, 10]
        ])
        target = torch.tensor([0, 2, 1])
        
        tracker.update(output, target)
        cm = tracker.get_confusion_matrix()
        
        # Rows = Target, Cols = Pred
        # [0,0] -> 1 (Correct)
        # [1,2] -> 1 (Target 1, Pred 2) -> incorrect
        # [2,1] -> 1 (Target 2, Pred 1) -> incorrect
        
        self.assertEqual(cm[0, 0], 1)
        self.assertEqual(cm[1, 2], 1) # Target 1 was predicted as 2 (wait, output[1] is [0,10,0] -> pred 1. Target is 2. So Target 2, Pred 1)
        # My setup above: 
        # 1st: Pred 0, Target 0 -> cm[0,0] += 1
        # 2nd: Pred 1, Target 2 -> cm[2,1] += 1
        # 3rd: Pred 2, Target 1 -> cm[1,2] += 1
        
        self.assertEqual(cm[2, 1], 1)
        self.assertEqual(cm[1, 2], 1)

if __name__ == '__main__':
    unittest.main()
