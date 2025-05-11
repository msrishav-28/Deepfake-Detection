# fusion/feature_fusion/attention_fusion.py
import torch
import torch.nn as nn
from ...models.base_model import BaseModel

class AttentionFusion(BaseModel):
    """Attention-based feature fusion"""
    
    def __init__(self, models, feature_dims, fusion_dim=512, num_classes=1):
        """
        Args:
            models: List of models
            feature_dims: List of feature dimensions
            fusion_dim: Fusion dimension
            num_classes: Number of classes
        """
        super().__init__()
        
        self.models = nn.ModuleList(models)
        
        # Feature projection layers
        self.projections = nn.ModuleList([
            nn.Linear(dim, fusion_dim) for dim in feature_dims
        ])
        
        # Cross-model attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            dropout=0.1
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim // 2, num_classes),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """Forward pass"""
        # Extract features from each model
        features = []
        for i, model in enumerate(self.models):
            feature = model.extract_features(x)
            projected = self.projections[i](feature)
            features.append(projected)
        
        # Stack features [batch_size, num_models, fusion_dim]
        stacked = torch.stack(features, dim=1)
        batch_size, num_models, dim = stacked.shape
        
        # Reshape for multi-head attention [num_models, batch_size, fusion_dim]
        reshaped = stacked.transpose(0, 1)
        
        # Apply cross-model attention
        attn_output, _ = self.cross_attention(
            query=reshaped,
            key=reshaped,
            value=reshaped
        )
        
        # Average across models
        fused = attn_output.mean(dim=0)
        
        # Classify
        output = self.classifier(fused)
        
        return output
    
    def extract_features(self, x):
        """Extract features"""
        # Extract features from each model
        features = []
        for i, model in enumerate(self.models):
            feature = model.extract_features(x)
            projected = self.projections[i](feature)
            features.append(projected)
        
        # Stack features [batch_size, num_models, fusion_dim]
        stacked = torch.stack(features, dim=1)
        batch_size, num_models, dim = stacked.shape
        
        # Reshape for multi-head attention [num_models, batch_size, fusion_dim]
        reshaped = stacked.transpose(0, 1)
        
        # Apply cross-model attention
        attn_output, _ = self.cross_attention(
            query=reshaped,
            key=reshaped,
            value=reshaped
        )
        
        # Average across models
        fused = attn_output.mean(dim=0)
        
        return fused