# training/losses/contrastive_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """Contrastive Loss"""
    
    def __init__(self, margin=1.0):
        """
        Args:
            margin: Margin
        """
        super().__init__()
        
        self.margin = margin
    
    def forward(self, outputs1, outputs2, targets):
        """
        Args:
            outputs1: Feature outputs 1
            outputs2: Feature outputs 2
            targets: Ground truth labels (1 for same, 0 for different)
            
        Returns:
            Loss
        """
        # Calculate Euclidean distance
        distances = F.pairwise_distance(outputs1, outputs2)
        
        # Calculate loss
        loss = targets * distances.pow(2) + (1 - targets) * F.relu(self.margin - distances).pow(2)
        
        return loss.mean()