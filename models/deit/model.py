# models/deit/model.py
import torch
import torch.nn as nn
import math
from einops import rearrange
from einops.layers.torch import Rearrange
from ..base_model import BaseModel
from ..vit.blocks import TransformerBlock
from ..vit.embedding import PatchEmbedding
from .distillation import DistillationToken

class DeiT(BaseModel):
    """Data-efficient Image Transformer (DeiT) model"""
    
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
        attn_dropout=0.0,
        distillation=True
    ):
        """
        Args:
            img_size: Input image size
            patch_size: Patch size
            in_channels: Number of input channels
            num_classes: Number of classes
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dim ratio
            dropout: Dropout rate
            attn_dropout: Attention dropout rate
            distillation: Whether to use distillation token
        """
        super().__init__()
        
        # Save params
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.distillation = distillation
        
        # Calculate number of patches
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Distillation token
        self.dist_token = None
        if distillation:
            self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embedding
        num_tokens = 1 + (1 if distillation else 0)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + num_tokens, embed_dim))
        
        # Dropout
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout
            )
            for _ in range(depth)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Distillation head
        self.head_dist = None
        if distillation:
            self.head_dist = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        # Initialize patch_embed
        nn.init.normal_(self.cls_token, std=0.02)
        if self.dist_token is not None:
            nn.init.normal_(self.dist_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Initialize other parameters
        self.apply(self._init_layer_weights)
    
    def _init_layer_weights(self, m):
        """Initialize layer weights"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x):
        """Forward pass through the transformer blocks"""
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token
        cls_token = self.cls_token.expand(B, -1, -1)
        
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            # Add distillation token
            dist_token = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_token, dist_token, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply layer normalization
        x = self.norm(x)
        
        if self.dist_token is None:
            return x[:, 0]  # Return class token only
        else:
            return x[:, 0], x[:, 1]  # Return class and distillation tokens
    
    def forward(self, x):
        """Forward pass"""
        if self.dist_token is None:
            x = self.forward_features(x)
            x = self.head(x)
            return torch.sigmoid(x)
        else:
            x, x_dist = self.forward_features(x)
            x = self.head(x)
            x_dist = self.head_dist(x_dist)
            # During inference, return the average of both outputs
            if not self.training:
                return torch.sigmoid((x + x_dist) / 2)
            else:
                return torch.sigmoid(x), torch.sigmoid(x_dist)
    
    def extract_features(self, x):
        """Extract features for feature fusion"""
        if self.dist_token is None:
            return self.forward_features(x)
        else:
            x, _ = self.forward_features(x)
            return x