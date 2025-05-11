# models/vit/embedding.py
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    """Patch embedding layer"""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        """
        Args:
            img_size: Input image size
            patch_size: Patch size
            in_channels: Number of input channels
            embed_dim: Embedding dimension
        """
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch projection
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Rearrange patches to sequence
        self.rearrange = Rearrange('b c h w -> b (h w) c')
    
    def forward(self, x):
        """Forward pass"""
        # Project patches
        x = self.proj(x)
        
        # Rearrange to sequence
        x = self.rearrange(x)
        
        return x