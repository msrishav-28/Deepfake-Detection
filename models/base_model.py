# models/base_model.py
import torch
import torch.nn as nn

class BaseModel(nn.Module):
    """Base model class with common methods"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """Forward pass"""
        raise NotImplementedError
    
    def extract_features(self, x):
        """Extract features for feature fusion"""
        raise NotImplementedError
    
    def save_checkpoint(self, path, optimizer=None, epoch=0, metadata=None):
        """Save model checkpoint"""
        checkpoint = {
            "model": self.state_dict(),
            "epoch": epoch
        }
        
        if optimizer is not None:
            checkpoint["optimizer"] = optimizer.state_dict()
            
        if metadata is not None:
            checkpoint["metadata"] = metadata
            
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path, optimizer=None, device="cuda"):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        
        self.load_state_dict(checkpoint["model"])
        
        if optimizer is not None and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            
        epoch = checkpoint.get("epoch", 0)
        metadata = checkpoint.get("metadata", None)
        
        return epoch, metadata