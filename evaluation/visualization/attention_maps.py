# evaluation/visualization/attention_maps.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def get_attention_maps(model, image, layer_idx=-1, head_idx=None):
    """
    Get attention maps from transformer model
    
    Args:
        model: Transformer model
        image: Input image tensor
        layer_idx: Layer index
        head_idx: Head index
        
    Returns:
        Attention maps
    """
    # Forward pass with hooks
    attention_maps = []
    
    def hook_fn(module, input, output):
        # Get attention maps
        # output: (batch_size, num_heads, seq_len, seq_len)
        attention_maps.append(output.detach())
    
    # Register hooks
    hooks = []
    if isinstance(model, torch.nn.Module):
        for name, module in model.named_modules():
            if 'attn' in name and 'attn_drop' not in name:
                hooks.append(module.register_forward_hook(hook_fn))
    
    # Forward pass
    with torch.no_grad():
        model(image.unsqueeze(0))
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Get attention maps
    if layer_idx >= 0 and layer_idx < len(attention_maps):
        attn_map = attention_maps[layer_idx]
    else:
        attn_map = attention_maps[-1]
    
    # Get specific head
    if head_idx is not None:
        attn_map = attn_map[0, head_idx]
    else:
        attn_map = attn_map[0].mean(dim=0)
    
    return attn_map.cpu().numpy()

def visualize_attention_maps(model, image, layer_idx=-1, head_idx=None, save_path=None):
    """
    Visualize attention maps
    
    Args:
        model: Transformer model
        image: Input image tensor
        layer_idx: Layer index
        head_idx: Head index
        save_path: Path to save visualization
        
    Returns:
        Figure
    """
    # Get attention maps
    attn_map = get_attention_maps(model, image, layer_idx, head_idx)
    
    # Create figure
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot original image
    img = image.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    axs[0].imshow(img)
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    # Plot attention map
    im = axs[1].imshow(attn_map, cmap='viridis')
    axs[1].set_title('Attention Map')
    plt.colorbar(im, ax=axs[1])
    
    # Save figure
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig