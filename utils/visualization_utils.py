import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

def tensor_to_image(tensor):
    """
    Convert tensor to image
    
    Args:
        tensor: PyTorch tensor
        
    Returns:
        NumPy array
    """
    # Clone tensor
    tensor = tensor.clone().detach()
    
    # Move to CPU
    tensor = tensor.cpu().squeeze()
    
    # Convert to NumPy
    if tensor.ndim == 3:
        # Image tensor
        npimg = tensor.numpy()
        
        # Convert from CxHxW to HxWxC
        npimg = np.transpose(npimg, (1, 2, 0))
    else:
        # Grayscale tensor
        npimg = tensor.numpy()
    
    # Denormalize if needed
    if npimg.max() <= 1.0:
        npimg = (npimg * 255).astype(np.uint8)
    
    return npimg

def plot_images(images, titles=None, figsize=(12, 8), rows=None, cols=None, save_path=None):
    """
    Plot multiple images
    
    Args:
        images: List of images
        titles: List of titles
        figsize: Figure size
        rows: Number of rows
        cols: Number of columns
        save_path: Path to save figure
        
    Returns:
        Figure
    """
    # Set default titles
    if titles is None:
        titles = [f"Image {i+1}" for i in range(len(images))]
    
    # Set default rows and columns
    if rows is None and cols is None:
        cols = min(4, len(images))
        rows = (len(images) + cols - 1) // cols
    elif rows is None:
        rows = (len(images) + cols - 1) // cols
    elif cols is None:
        cols = (len(images) + rows - 1) // rows
    
    # Create figure
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    
    # Flatten axes if needed
    if rows == 1 and cols == 1:
        axs = np.array([axs])
    elif rows == 1 or cols == 1:
        axs = axs.flatten()
    
    # Plot images
    for i, (image, title) in enumerate(zip(images, titles)):
        if i < len(axs):
            # Convert tensor to image if needed
            if hasattr(image, 'cpu'):
                image = tensor_to_image(image)
            
            # Plot image
            axs[i].imshow(image)
            axs[i].set_title(title)
            axs[i].axis('off')
    
    # Hide empty subplots
    for i in range(len(images), rows * cols):
        if i < len(axs):
            axs[i].axis('off')
    
    # Set tight layout
    plt.tight_layout()
    
    # Save figure
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig

def overlay_mask(image, mask, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlay mask on image
    
    Args:
        image: Image
        mask: Mask
        alpha: Alpha value for blending
        colormap: Colormap for mask
        
    Returns:
        Overlayed image
    """
    # Convert tensor to image if needed
    if hasattr(image, 'cpu'):
        image = tensor_to_image(image)
    
    if hasattr(mask, 'cpu'):
        mask = tensor_to_image(mask)
    
    # Normalize mask
    mask = mask.astype(np.float32)
    mask = mask - mask.min()
    mask = mask / mask.max()
    
    # Create color mask
    color_mask = cv2.applyColorMap((mask * 255).astype(np.uint8), colormap)
    
    # Convert BGR to RGB
    color_mask = cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB)
    
    # Resize mask to match image
    if color_mask.shape[:2] != image.shape[:2]:
        color_mask = cv2.resize(color_mask, (image.shape[1], image.shape[0]))
    
    # Overlay mask on image
    overlay = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)
    
    return overlay