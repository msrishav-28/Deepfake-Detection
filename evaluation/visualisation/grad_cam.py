# evaluation/visualization/grad_cam.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

class GradCAM:
    """Grad-CAM implementation for transformer models"""
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: Model
            target_layer: Target layer
        """
        self.model = model
        self.target_layer = target_layer
        self.hooks = []
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        # Forward hook
        def forward_hook(module, input, output):
            self.activations = output
        
        # Backward hook
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Register hooks
        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_backward_hook(backward_hook))
    
    def remove_hooks(self):
        """Remove hooks"""
        for hook in self.hooks:
            hook.remove()
    
    def __call__(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM
        
        Args:
            input_tensor: Input tensor
            class_idx: Class index
            
        Returns:
            heatmap: Grad-CAM heatmap
        """
        # Forward pass
        self.model.zero_grad()
        output = self.model(input_tensor.unsqueeze(0))
        
        # Get class index
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        # Backward pass
        output[0, class_idx].backward()
        
        # Get gradients and activations
        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]
        
        # Calculate weights
        weights = np.mean(gradients, axis=(1, 2))
        
        # Generate heatmap
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            heatmap += w * activations[i]
        
        # ReLU
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize
        heatmap = heatmap - np.min(heatmap)
        heatmap = heatmap / np.max(heatmap)
        
        return heatmap

def visualize_grad_cam(model, image, target_layer, class_idx=None, save_path=None):
    """
    Visualize Grad-CAM
    
    Args:
        model: Model
        image: Input image tensor
        target_layer: Target layer
        class_idx: Class index
        save_path: Path to save visualization
        
    Returns:
        Figure
    """
    # Create Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # Generate heatmap
    heatmap = grad_cam(image, class_idx)
    
    # Convert to uint8
    heatmap = np.uint8(255 * heatmap)
    
    # Apply colormap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Convert image to numpy array
    if isinstance(image, torch.Tensor):
        img = image.permute(1, 2, 0).cpu().numpy()
        img = np.uint8(255 * img)
    else:
        img = np.array(image)
    
    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Overlay heatmap on image
    superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
    # Create figure
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot original image
    axs[0].imshow(img)
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    # Plot heatmap
    axs[1].imshow(heatmap)
    axs[1].set_title('Grad-CAM Heatmap')
    axs[1].axis('off')
    
    # Plot superimposed image
    axs[2].imshow(superimposed)
    axs[2].set_title('Superimposed')
    axs[2].axis('off')
    
    # Save figure
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    # Remove hooks
    grad_cam.remove_hooks()
    
    return fig