# evaluation/metrics/roc_metrics.py
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

def calculate_roc_metrics(targets, predictions):
    """
    Calculate ROC metrics
    
    Args:
        targets: Ground truth labels
        predictions: Model predictions
        
    Returns:
        fpr: False positive rate
        tpr: True positive rate
        thresholds: Thresholds
        auc: Area under ROC curve
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(targets, predictions)
    
    # Calculate AUC
    auc = roc_auc_score(targets, predictions)
    
    return fpr, tpr, thresholds, auc

def calculate_eer(targets, predictions):
    """
    Calculate Equal Error Rate (EER)
    
    Args:
        targets: Ground truth labels
        predictions: Model predictions
        
    Returns:
        eer: Equal Error Rate
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(targets, predictions)
    
    # Calculate EER
    fnr = 1 - tpr
    
    # Find the threshold where FPR = FNR
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    
    return float(eer)