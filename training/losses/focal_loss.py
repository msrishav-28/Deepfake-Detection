# training/losses/focal_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    """Binary Focal Loss"""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Weighting factor
            gamma: Focusing parameter
            reduction: Reduction method
        """
        super().__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predicted probabilities (N, 1)
            targets: Ground truth labels (N, 1)
            
        Returns:
            Loss
        """
        # Get binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        
        # Get pt
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        # Calculate loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss