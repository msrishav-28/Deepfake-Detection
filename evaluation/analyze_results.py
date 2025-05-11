# evaluation/analyze_results.py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
from .visualization import (
    plot_roc_curve, plot_confusion_matrix, plot_metrics
)

class ModelEvaluator:
    """
    Class for model evaluation and analysis
    """
    
    def __init__(self, model, device='cuda'):
        """
        Initialize ModelEvaluator
        
        Args:
            model: PyTorch model
            device: Device to run evaluation
        """
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.metrics = {}
    
    def evaluate(self, dataloader, criterion=None):
        """
        Evaluate model on a dataset
        
        Args:
            dataloader: PyTorch DataLoader
            criterion: Loss function
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Prepare lists for results
        all_preds = []
        all_targets = []
        all_probs = []
        running_loss = 0.0
        
        # Evaluate model
        with torch.no_grad():
            for inputs, targets in dataloader:
                # Move to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss
                if criterion is not None:
                    loss = criterion(outputs, targets)
                    running_loss += loss.item() * inputs.size(0)
                
                # Get predictions
                if outputs.shape[1] > 1:  # Multi-class
                    probs = torch.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)
                else:  # Binary
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).long()
                
                # Store results
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
        
        # Concatenate results
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        all_probs = np.concatenate(all_probs)
        
        # Calculate metrics
        metrics = {}
        
        # Loss
        if criterion is not None:
            metrics['loss'] = running_loss / len(dataloader.dataset)
        
        # Classification metrics
        metrics['accuracy'] = accuracy_score(all_targets, all_preds)
        
        # Handle binary and multi-class scenarios
        if len(np.unique(all_targets)) <= 2:
            # Binary classification
            metrics['precision'] = precision_score(all_targets, all_preds, zero_division=0)
            metrics['recall'] = recall_score(all_targets, all_preds, zero_division=0)
            metrics['f1'] = f1_score(all_targets, all_preds, zero_division=0)
            
            if all_probs.shape[1] == 1:
                metrics['roc_auc'] = roc_auc_score(all_targets, all_probs.flatten())
            else:
                metrics['roc_auc'] = roc_auc_score(all_targets, all_probs[:, 1])
        else:
            # Multi-class classification
            metrics['precision'] = precision_score(all_targets, all_preds, average='macro', zero_division=0)
            metrics['recall'] = recall_score(all_targets, all_preds, average='macro', zero_division=0)
            metrics['f1'] = f1_score(all_targets, all_preds, average='macro', zero_division=0)
            
            # Convert targets to one-hot for multi-class AUC
            n_classes = all_probs.shape[1]
            targets_one_hot = np.zeros((all_targets.size, n_classes))
            targets_one_hot[np.arange(all_targets.size), all_targets] = 1
            
            metrics['roc_auc'] = roc_auc_score(targets_one_hot, all_probs, average='macro', multi_class='ovr')
        
        # Store predictions and targets for further analysis
        self.predictions = all_preds
        self.targets = all_targets
        self.probabilities = all_probs
        self.metrics = metrics
        
        return metrics
    
    def generate_classification_report(self, class_names=None):
        """
        Generate classification report
        
        Args:
            class_names: List of class names
            
        Returns:
            Classification report
        """
        return classification_report(
            self.targets, 
            self.predictions, 
            target_names=class_names
        )
    
    def visualize_results(self, class_names=None, output_dir=None):
        """
        Visualize evaluation results
        
        Args:
            class_names: List of class names
            output_dir: Directory to save visualizations
            
        Returns:
            Dictionary of figures
        """
        figures = {}
        
        # Determine if binary or multi-class
        n_classes = len(np.unique(self.targets))
        
        # Convert targets to one-hot for ROC curve if multi-class
        if n_classes > 2:
            targets_one_hot = np.zeros((self.targets.size, n_classes))
            targets_one_hot[np.arange(self.targets.size), self.targets] = 1
            targets_for_roc = targets_one_hot
        else:
            targets_for_roc = self.targets
        
        # Plot ROC curve
        roc_path = f"{output_dir}/roc_curve.png" if output_dir else None
        figures['roc_curve'] = plot_roc_curve(
            targets_for_roc, 
            self.probabilities, 
            class_names=class_names,
            save_path=roc_path
        )
        
        # Plot confusion matrix
        cm_path = f"{output_dir}/confusion_matrix.png" if output_dir else None
        figures['confusion_matrix'] = plot_confusion_matrix(
            self.targets, 
            self.predictions, 
            class_names=class_names,
            save_path=cm_path
        )
        
        return figures

def analyze_model_performance(model, test_loader, criterion=None, class_names=None, output_dir=None):
    """
    Analyze model performance
    
    Args:
        model: PyTorch model
        test_loader: Test DataLoader
        criterion: Loss function
        class_names: List of class names
        output_dir: Directory to save visualizations
        
    Returns:
        Dictionary of metrics and figures
    """
    # Create evaluator
    evaluator = ModelEvaluator(model, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Evaluate model
    metrics = evaluator.evaluate(test_loader, criterion)
    
    # Generate classification report
    report = evaluator.generate_classification_report(class_names)
    
    # Visualize results
    figures = evaluator.visualize_results(class_names, output_dir)
    
    return {
        'metrics': metrics,
        'report': report,
        'figures': figures
    }

def compare_models(models, model_names, test_loader, criterion=None, class_names=None, output_dir=None):
    """
    Compare multiple models
    
    Args:
        models: List of PyTorch models
        model_names: List of model names
        test_loader: Test DataLoader
        criterion: Loss function
        class_names: List of class names
        output_dir: Directory to save visualizations
        
    Returns:
        DataFrame with model comparison
    """
    # Prepare list for results
    results = []
    
    # Evaluate each model
    for model, name in zip(models, model_names):
        # Create evaluator
        evaluator = ModelEvaluator(model, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Evaluate model
        metrics = evaluator.evaluate(test_loader, criterion)
        
        # Add model name
        metrics['model'] = name
        
        # Append to results
        results.append(metrics)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Plot comparison
    if output_dir:
        # Select only metric columns
        metric_cols = [col for col in df.columns if col != 'model']
        
        # Create figure
        fig, axes = plt.subplots(len(metric_cols), 1, figsize=(12, 4 * len(metric_cols)))
        
        # Handle single metric case
        if len(metric_cols) == 1:
            axes = [axes]
        
        # Plot each metric
        for i, metric in enumerate(metric_cols):
            axes[i].bar(df['model'], df[metric])
            axes[i].set_xlabel('Model')
            axes[i].set_ylabel(metric)
            axes[i].set_title(f'{metric} Comparison')
            
            # Add value labels
            for j, val in enumerate(df[metric]):
                axes[i].text(j, val, f'{val:.4f}', ha='center', va='bottom')
            
            # Set y-limit
            axes[i].set_ylim([0, max(df[metric]) * 1.1])
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{output_dir}/model_comparison.png", bbox_inches='tight')
    
    return df