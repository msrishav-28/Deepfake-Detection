# models/model_zoo/model_factory.py
from ..vit.model import ViT
from ..deit.model import DeiT
from ..swin.model import SwinTransformer
import inspect

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
    
    # Create Swin Transformer with filtered parameters
    elif model_type == "swin":
        # Get valid parameters for SwinTransformer
        valid_params = inspect.signature(SwinTransformer.__init__).parameters.keys()
        
        # Filter kwargs to only include valid parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        
        print(f"Original kwargs: {list(kwargs.keys())}")
        print(f"Filtered kwargs: {list(filtered_kwargs.keys())}")
        print(f"Removed params: {set(kwargs.keys()) - set(filtered_kwargs.keys())}")
        
        return SwinTransformer(**filtered_kwargs)
    
    # Unknown model type
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_model_params(model_type, num_classes=2, img_size=224):
    """Get default parameters for each model type"""
    
    if model_type == "vit":
        return {
            'num_classes': num_classes,
            'img_size': img_size,
            'patch_size': 16,
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
            'mlp_ratio': 4.0,
            'drop_rate': 0.1
        }
    
    elif model_type == "deit":
        return {
            'num_classes': num_classes,
            'img_size': img_size,
            'patch_size': 16,
            'embed_dim': 384,
            'depth': 12,
            'num_heads': 6,
            'mlp_ratio': 4.0,
            'drop_rate': 0.1
        }
    
    elif model_type == "swin":
        return {
            'num_classes': num_classes,
            'img_size': img_size,
            'patch_size': 4,
            'embed_dim': 96,
            'depths': [2, 2, 6, 2],  # Note: 'depths' not 'depth'
            'num_heads': [3, 6, 12, 24],
            'window_size': 7,
            'drop_rate': 0.1,
            'drop_path_rate': 0.2
        }
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")