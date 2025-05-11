# fusion/ensemble/stacking.py
import torch
import torch.nn as nn
from ...models.base_model import BaseModel

class StackingEnsemble(BaseModel):
    """Stacking ensemble"""
    
    def __init__(self, models, meta_model=None):
        """
        Args:
            models: List of models
            meta_model: Meta model for stacking
        """
        super().__init__()
        
        self.models = nn.ModuleList(models)
        
        # Create meta model if not provided
        if meta_model is None:
            num_models = len(models)
            
            # Create simple meta model
            self.meta_model = nn.Sequential(
                nn.Linear(num_models, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        else:
            self.meta_model = meta_model
        
    def forward(self, x):
        """Forward pass"""
        # Get predictions from each model
        preds = []
        for model in self.models:
            with torch.no_grad():
                pred = model(x)
            preds.append(pred)
        
        # Stack predictions
        stacked_preds = torch.cat(preds, dim=1)
        
        # Apply meta model
        return self.meta_model(stacked_preds)
    
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