# training/losses/triplet_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """Triplet Loss"""
    
    def __init__(self, margin=1.0):
        """
        Args:
            margin: Margin
        """
        super().__init__()
        
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: Anchor feature
            positive: Positive feature
            negative: Negative feature
            
        Returns:
            Loss
        """
        # Calculate distances
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        
        # Calculate loss
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        return loss.mean()