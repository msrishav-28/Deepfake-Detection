# models/swin/model.py - Fixed imports section
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np

from .blocks import SwinTransformerBlock, BasicLayer, PatchMerging

from einops import rearrange
from ..base_model import BaseModel

class SwinTransformer(BaseModel):
    """Swin Transformer model"""
    
    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_channels=3,
        num_classes=1,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        dropout=0.0,
        attn_dropout=0.0,
        patch_norm=True
    ):
        """
        Args:
            img_size: Input image size
            patch_size: Patch size
            in_channels: Number of input channels
            num_classes: Number of classes
            embed_dim: Embedding dimension
            depths: Depths of each Swin stage
            num_heads: Number of attention heads in each Swin stage
            window_size: Window size
            mlp_ratio: MLP hidden dim ratio
            dropout: Dropout rate
            attn_dropout: Attention dropout rate
            patch_norm: Whether to add normalization after patch embedding
        """
        super().__init__()
        
        # Save params
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_layers = len(depths)
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm if patch_norm else None
        )
        
        # Get patch grid size and number of patches
        self.patches_resolution = self.patch_embed.patches_resolution
        self.num_patches = self.patch_embed.num_patches
        
        # Position dropout
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]
        
        # Build Swin layers
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(
                BasicLayer(
                    dim=int(embed_dim * 2 ** i),
                    input_resolution=(
                        self.patches_resolution[0] // (2 ** i),
                        self.patches_resolution[1] // (2 ** i)
                    ),
                    depth=depths[i],
                    num_heads=num_heads[i],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    drop_path=dpr[sum(depths[:i]):sum(depths[:i+1])],
                    downsample=PatchMerging if (i < self.num_layers - 1) else None
                )
            )
        
        # Layer normalization
        self.norm = nn.LayerNorm(int(embed_dim * 2 ** (self.num_layers - 1)))
        
        # Classification head
        self.head = nn.Linear(int(embed_dim * 2 ** (self.num_layers - 1)), num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        def _init_fn(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        
        self.apply(_init_fn)
    
    def forward_features(self, x):
        """Forward pass through the Swin transformer blocks"""
        # Patch embedding
        x = self.patch_embed(x)
        
        # Position dropout
        x = self.pos_drop(x)
        
        # Apply Swin layers
        for layer in self.layers:
            x = layer(x)
        
        # Layer normalization
        x = self.norm(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        return x
    
    def forward(self, x):
        """Forward pass"""
        x = self.forward_features(x)
        x = self.head(x)
        return torch.sigmoid(x)
    
    def extract_features(self, x):
        """Extract features for feature fusion"""
        return self.forward_features(x)


class PatchEmbed(nn.Module):
    """Patch embedding for Swin Transformer"""
    
    def __init__(self, img_size=224, patch_size=4, in_channels=3, embed_dim=96, norm_layer=None):
        """
        Args:
            img_size: Input image size
            patch_size: Patch size
            in_channels: Number of input channels
            embed_dim: Embedding dimension
            norm_layer: Normalization layer
        """
        super().__init__()
        
        # Handle img_size as tuple
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        
        # Handle patch_size as tuple
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
            
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        
        # Patch projection
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Normalization
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    
    def forward(self, x):
        """Forward pass"""
        B, C, H, W = x.shape
        
        # Check image size
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})"
        
        # Patch projection
        x = self.proj(x).flatten(2).transpose(1, 2)
        
        # Normalization
        x = self.norm(x)
        
        return x