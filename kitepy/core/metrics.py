"""
Metrics module for training evaluation.

Provides accuracy, precision, recall, F1, and more.
"""

import torch
from typing import Dict, Any, Optional, List
from collections import defaultdict


class MetricTracker:
    """
    Track multiple metrics during training.
    
    Example:
        tracker = MetricTracker()
        tracker.update(output, target)
        metrics = tracker.compute()
    """
    
    def __init__(self, num_classes: Optional[int] = None):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.correct = 0
        self.total = 0
        self.loss_sum = 0.0
        self.loss_count = 0
        
        # For precision/recall/F1
        self.true_positives = defaultdict(int)
        self.false_positives = defaultdict(int)
        self.false_negatives = defaultdict(int)
        
        # For confusion matrix
        self.predictions = []
        self.targets = []
    
    def update(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        loss: Optional[float] = None
    ):
        """
        Update metrics with batch results.
        
        Args:
            output: Model output logits (B, num_classes)
            target: Ground truth labels (B,)
            loss: Optional loss value
        """
        with torch.no_grad():
            pred = output.argmax(dim=1)
            
            # Accuracy
            self.correct += pred.eq(target).sum().item()
            self.total += target.size(0)
            
            # Loss
            if loss is not None:
                self.loss_sum += loss
                self.loss_count += 1
            
            # Per-class metrics
            for p, t in zip(pred.cpu().numpy(), target.cpu().numpy()):
                if p == t:
                    self.true_positives[t] += 1
                else:
                    self.false_positives[p] += 1
                    self.false_negatives[t] += 1
            
            # Store for confusion matrix
            self.predictions.extend(pred.cpu().numpy().tolist())
            self.targets.extend(target.cpu().numpy().tolist())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dict with accuracy, precision, recall, f1, loss
        """
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = 100.0 * self.correct / self.total if self.total > 0 else 0.0
        
        # Loss
        metrics['loss'] = self.loss_sum / self.loss_count if self.loss_count > 0 else 0.0
        
        # Get all classes
        all_classes = set(self.true_positives.keys()) | set(self.false_positives.keys()) | set(self.false_negatives.keys())
        
        if all_classes:
            # Per-class precision, recall, F1
            precisions = []
            recalls = []
            f1s = []
            
            for c in all_classes:
                tp = self.true_positives[c]
                fp = self.false_positives[c]
                fn = self.false_negatives[c]
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)
            
            # Macro averages
            metrics['precision'] = sum(precisions) / len(precisions)
            metrics['recall'] = sum(recalls) / len(recalls)
            metrics['f1'] = sum(f1s) / len(f1s)
        
        return metrics
    
    def get_confusion_matrix(self) -> torch.Tensor:
        """Get confusion matrix."""
        num_classes = max(max(self.predictions) + 1, max(self.targets) + 1) if self.predictions else 0
        matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
        
        for p, t in zip(self.predictions, self.targets):
            matrix[t, p] += 1
        
        return matrix


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple = (1,)) -> List[float]:
    """
    Compute top-k accuracy.
    
    Args:
        output: Model output logits (B, num_classes)
        target: Ground truth labels (B,)
        topk: Tuple of k values
    
    Returns:
        List of accuracies for each k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        
        return res


def precision_recall_f1(
    output: torch.Tensor,
    target: torch.Tensor,
    average: str = 'macro'
) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 score.
    
    Args:
        output: Model output logits
        target: Ground truth labels
        average: 'macro', 'micro', or 'weighted'
    
    Returns:
        Dict with precision, recall, f1
    """
    pred = output.argmax(dim=1)
    
    num_classes = output.size(1)
    
    precisions = []
    recalls = []
    f1s = []
    supports = []
    
    for c in range(num_classes):
        tp = ((pred == c) & (target == c)).sum().item()
        fp = ((pred == c) & (target != c)).sum().item()
        fn = ((pred != c) & (target == c)).sum().item()
        support = (target == c).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)
    
    if average == 'macro':
        return {
            'precision': sum(precisions) / len(precisions),
            'recall': sum(recalls) / len(recalls),
            'f1': sum(f1s) / len(f1s),
        }
    elif average == 'micro':
        total_tp = sum(((pred == c) & (target == c)).sum().item() for c in range(num_classes))
        total_fp = sum(((pred == c) & (target != c)).sum().item() for c in range(num_classes))
        total_fn = sum(((pred != c) & (target == c)).sum().item() for c in range(num_classes))
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {'precision': precision, 'recall': recall, 'f1': f1}
    elif average == 'weighted':
        total = sum(supports)
        return {
            'precision': sum(p * s for p, s in zip(precisions, supports)) / total if total > 0 else 0.0,
            'recall': sum(r * s for r, s in zip(recalls, supports)) / total if total > 0 else 0.0,
            'f1': sum(f * s for f, s in zip(f1s, supports)) / total if total > 0 else 0.0,
        }
    else:
        raise ValueError(f"Unknown average: {average}")


__all__ = [
    'MetricTracker',
    'accuracy',
    'precision_recall_f1',
]
