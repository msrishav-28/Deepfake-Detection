# evaluation/metrics/classification_metrics.py
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from .roc_metrics import calculate_eer

def calculate_metrics(predictions, targets, threshold=0.5):
    """
    Calculate classification metrics
    
    Args:
        predictions: Model predictions
        targets: Ground truth labels
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    # Convert predictions to binary
    binary_preds = (predictions > threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(targets, binary_preds)
    precision = precision_score(targets, binary_preds, zero_division=0)
    recall = recall_score(targets, binary_preds, zero_division=0)
    f1 = f1_score(targets, binary_preds, zero_division=0)
    auc = roc_auc_score(targets, predictions)
    eer = calculate_eer(targets, predictions)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(auc),
        'eer': float(eer)
    }