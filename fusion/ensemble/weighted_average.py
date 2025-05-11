# fusion/ensemble/weighted_average.py
import torch
import torch.nn as nn
from ...models.base_model import BaseModel

class WeightedAverage(BaseModel):
    """Weighted average ensemble"""
    
    def __init__(self, models, weights=None):
        """
        Args:
            models: List of models
            weights: List of weights
        """
        super().__init__()
        
        self.models = nn.ModuleList(models)
        
        # Set weights
        if weights is None:
            weights = torch.ones(len(models))
        else:
            weights = torch.tensor(weights)
            
        # Normalize weights
        weights = weights / weights.sum()
        self.register_buffer("weights", weights)
        
    def forward(self, x):
        """Forward pass"""
        # Get predictions from each model
        preds = []
        for i, model in enumerate(self.models):
            with torch.no_grad():
                pred = model(x)
            preds.append(pred * self.weights[i])
        
        # Sum weighted predictions
        return torch.stack(preds).sum(dim=0)
    
    def extract_features(self, x):
        """Extract features"""
        # Get features from each model
        features = []
        for i, model in enumerate(self.models):
            with torch.no_grad():
                feature = model.extract_features(x)
            features.append(feature)
        
        # Concatenate features
        features = torch.cat(features, dim=1)
        
        return features