# evaluation/metrics/confusion_matrix.py
import numpy as np
from sklearn.metrics import confusion_matrix

def calculate_confusion_matrix(targets, predictions, threshold=0.5):
    """
    Calculate confusion matrix
    
    Args:
        targets: Ground truth labels
        predictions: Model predictions
        threshold: Classification threshold
        
    Returns:
        cm: Confusion matrix
        cm_normalized: Normalized confusion matrix
    """
    # Convert predictions to binary
    binary_preds = (predictions > threshold).astype(int)
    
    # Calculate confusion matrix
    cm = confusion_matrix(targets, binary_preds)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    return cm, cm_normalized