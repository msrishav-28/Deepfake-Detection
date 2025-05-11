# models/model_zoo/model_factory.py
from ..vit.model import ViT
from ..deit.model import DeiT
from ..swin.model import SwinTransformer

def create_model(model_type, **kwargs):
    """
    Create model from type and parameters
    
    Args:
        model_type: Model type
        **kwargs: Model parameters
        
    Returns:
        Model instance
    """
    # Create Vision Transformer (ViT)
    if model_type == "vit":
        return ViT(**kwargs)
    
    # Create Data-efficient Image Transformer (DeiT)
    elif model_type == "deit":
        return DeiT(**kwargs)
    
    # Create Swin Transformer
    elif model_type == "swin":
        return SwinTransformer(**kwargs)
    
    # Unknown model type
    else:
        raise ValueError(f"Unknown model type: {model_type}")