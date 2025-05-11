# models/deit/distillation.py
import torch
import torch.nn as nn

class DistillationToken(nn.Module):
    """Distillation token for DeiT"""
    
    def __init__(self, dim):
        """
        Args:
            dim: Token dimension
        """
        super().__init__()
        
        self.token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.token, std=0.02)
    
    def forward(self, x):
        """Forward pass"""
        B = x.shape[0]
        return self.token.expand(B, -1, -1)