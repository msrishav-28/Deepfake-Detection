# fusion/feature_fusion/transformer_fusion.py
import torch
import torch.nn as nn
from ...models.base_model import BaseModel
from ...models.vit.blocks import TransformerBlock

class TransformerFusion(BaseModel):
    """Transformer-based feature fusion"""
    
    def __init__(self, models, feature_dims, fusion_dim=512, depth=2, num_heads=8, num_classes=1):
        """
        Args:
            models: List of models
            feature_dims: List of feature dimensions
            fusion_dim: Fusion dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            num_classes: Number of classes
        """
        super().__init__()
        
        self.models = nn.ModuleList(models)
        
        # Feature projection layers
        self.projections = nn.ModuleList([
            nn.Linear(dim, fusion_dim) for dim in feature_dims
        ])
        
        # Special token for fusion
        self.fusion_token = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        nn.init.normal_(self.fusion_token, std=0.02)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, len(models) + 1, fusion_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Dropout
        self.pos_drop = nn.Dropout(p=0.1)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=fusion_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                dropout=0.1,
                attn_dropout=0.1
            )
            for _ in range(depth)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(fusion_dim)
        
        # Classification head
        self.head = nn.Linear(fusion_dim, num_classes)
        
    def forward(self, x):
        """Forward pass"""
        B = x.shape[0]
        
        # Extract features from each model
        features = []
        for i, model in enumerate(self.models):
            feature = model.extract_features(x)
            projected = self.projections[i](feature)
            features.append(projected)
        
        # Stack features [batch_size, num_models, fusion_dim]
        stacked = torch.stack(features, dim=1)
        
        # Add fusion token
        fusion_token = self.fusion_token.expand(B, -1, -1)
        x = torch.cat((fusion_token, stacked), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply layer normalization
        x = self.norm(x)
        
        # Get fusion token
        x = x[:, 0]
        
        # Apply classification head
        x = self.head(x)
        
        return torch.sigmoid(x)
    
    def extract_features(self, x):
        """Extract features"""
        B = x.shape[0]
        
        # Extract features from each model
        features = []
        for i, model in enumerate(self.models):
            feature = model.extract_features(x)
            projected = self.projections[i](feature)
            features.append(projected)
        
        # Stack features [batch_size, num_models, fusion_dim]
        stacked = torch.stack(features, dim=1)
        
        # Add fusion token
        fusion_token = self.fusion_token.expand(B, -1, -1)
        x = torch.cat((fusion_token, stacked), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply layer normalization
        x = self.norm(x)
        
        # Get fusion token
        x = x[:, 0]
        
        return x