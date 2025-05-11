# fusion/feature_fusion/concat_fusion.py
import torch
import torch.nn as nn
from ...models.base_model import BaseModel

class ConcatFusion(BaseModel):
    """Feature concatenation fusion"""
    
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
        
        # Calculate total feature dimension
        total_dim = sum(feature_dims)
        
        # Create fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.BatchNorm1d(fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fusion_dim // 2, num_classes),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """Forward pass"""
        # Extract features from each model
        features = []
        for model in self.models:
            feature = model.extract_features(x)
            features.append(feature)
        
        # Concatenate features
        concat_features = torch.cat(features, dim=1)
        
        # Apply fusion layers
        output = self.fusion(concat_features)
        
        return output
    
    def extract_features(self, x):
        """Extract features"""
        # Extract features from each model
        features = []
        for model in self.models:
            feature = model.extract_features(x)
            features.append(feature)
        
        # Concatenate features
        concat_features = torch.cat(features, dim=1)
        
        return concat_features