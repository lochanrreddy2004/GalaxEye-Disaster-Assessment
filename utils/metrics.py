import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import confusion_matrix


class SegmentationMetrics:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def compute(self, preds, targets):
        preds = (preds > self.threshold).astype(np.uint8)
        targets = targets.astype(np.uint8)

        preds = preds.flatten()
        targets = targets.flatten()

        precision = precision_score(targets, preds, zero_division=0)
        recall = recall_score(targets, preds, zero_division=0)
        f1 = f1_score(targets, preds, zero_division=0)
        iou = jaccard_score(targets, preds, zero_division=0)

        cm = confusion_matrix(targets, preds)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou,
            'confusion_matrix': cm
        }