# evaluation/visualization/results_plots.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve

def plot_roc_curve(y_true, y_pred, class_names=None, save_path=None):
    """
    Plot ROC curve for multi-class classification
    
    Args:
        y_true: Ground truth labels (one-hot encoded for multi-class)
        y_pred: Predicted probabilities
        class_names: List of class names
        save_path: Path to save the plot
        
    Returns:
        Figure
    """
    plt.figure(figsize=(10, 8))
    
    # Check if binary or multi-class
    if len(y_pred.shape) == 1 or y_pred.shape[1] == 1:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    else:
        # Multi-class classification
        n_classes = y_pred.shape[1]
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(n_classes)]
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], lw=2, 
                     label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Set labels and title
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    
    # Save figure
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return plt.gcf()

def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=False, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        normalize: Whether to normalize the confusion matrix
        save_path: Path to save the plot
        
    Returns:
        Figure
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if required
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    # Set labels and title
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return plt.gcf()

def plot_metrics(train_metrics, val_metrics, metric_names=None, save_path=None):
    """
    Plot training metrics
    
    Args:
        train_metrics: Dictionary of training metrics
        val_metrics: Dictionary of validation metrics
        metric_names: List of metric names to plot
        save_path: Path to save the plot
        
    Returns:
        Figure
    """
    # Get metric names if not provided
    if metric_names is None:
        metric_names = train_metrics.keys()
    
    # Create figure with subplots
    n_metrics = len(metric_names)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics))
    
    # Handle single metric case
    if n_metrics == 1:
        axes = [axes]
    
    # Plot each metric
    for i, metric in enumerate(metric_names):
        if metric in train_metrics and metric in val_metrics:
            axes[i].plot(train_metrics[metric], label=f'Train {metric}')
            axes[i].plot(val_metrics[metric], label=f'Validation {metric}')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric)
            axes[i].set_title(f'{metric} vs. Epoch')
            axes[i].legend()
            axes[i].grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig