# models/swin/patch_merging.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchMerging(nn.Module):
    """Patch Merging Layer for Swin Transformer"""
    
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        """
        Args:
            input_resolution: Input resolution
            dim: Number of input channels
            norm_layer: Normalization layer
        """
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
    
    def forward(self, x):
        """
        Args:
            x: Input feature, shape (B, H*W, C)
        Returns:
            Merged feature with shape (B, H/2*W/2, 2*C)
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        
        x = x.view(B, H, W, C)
        
        # Merge patches
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        
        x = self.norm(x)
        x = self.reduction(x)
        
        return x