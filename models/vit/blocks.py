# models/vit/blocks.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """Multi-head attention"""
    
    def __init__(self, dim, num_heads=8, attn_dropout=0.0):
        """
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            attn_dropout: Attention dropout rate
        """
        super().__init__()
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Query, key, value projection
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        
        # Attention dropout
        self.attn_drop = nn.Dropout(attn_dropout)
        
        # Output projection
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        """Forward pass"""
        B, N, C = x.shape
        
        # Project query, key, value
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Calculate attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # Output projection
        x = self.proj(x)
        
        return x


class MLP(nn.Module):
    """MLP block"""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.0):
        """
        Args:
            in_features: Input features
            hidden_features: Hidden features
            out_features: Output features
            dropout: Dropout rate
        """
        super().__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(dropout)
    
    def forward(self, x):
        """Forward pass"""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer block"""
    
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0, attn_dropout=0.0):
        """
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dim ratio
            dropout: Dropout rate
            attn_dropout: Attention dropout rate
        """
        super().__init__()
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        
        # Multi-head attention
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout
        )
        
        # Dropout
        self.drop1 = nn.Dropout(dropout)
        
        # Layer normalization
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            dropout=dropout
        )
        
        # Dropout
        self.drop2 = nn.Dropout(dropout)
    
    def forward(self, x):
        """Forward pass"""
        # Attention block
        attn_out = self.attn(self.norm1(x))
        x = x + self.drop1(attn_out)
        
        # MLP block
        mlp_out = self.mlp(self.norm2(x))
        x = x + self.drop2(mlp_out)
        
        return x