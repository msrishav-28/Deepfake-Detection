# fusion/ensemble/simple_voting.py
import torch
import torch.nn as nn
from ...models.base_model import BaseModel

class SimpleVoting(BaseModel):
    """Simple voting ensemble"""
    
    def __init__(self, models, threshold=0.5):
        """
        Args:
            models: List of models
            threshold: Voting threshold
        """
        super().__init__()
        
        self.models = nn.ModuleList(models)
        self.threshold = threshold
        
    def forward(self, x):
        """Forward pass"""
        # Get predictions from each model
        preds = []
        for model in self.models:
            with torch.no_grad():
                pred = model(x)
            preds.append(pred)
        
        # Stack predictions
        preds = torch.stack(preds, dim=0)
        
        # Calculate votes
        votes = (preds > self.threshold).float().mean(dim=0)
        
        return votes
    
    def extract_features(self, x):
        """Extract features"""
        # Get features from each model
        features = []
        for model in self.models:
            with torch.no_grad():
                feature = model.extract_features(x)
            features.append(feature)
        
        # Concatenate features
        features = torch.cat(features, dim=1)
        
        return features