# evaluation/visualization/feature_visualization.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def extract_features(model, dataloader, device):
    """
    Extract features from model
    
    Args:
        model: Model
        dataloader: Dataloader
        device: Device
        
    Returns:
        features: Extracted features
        labels: Ground truth labels
    """
    model.eval()
    
    # Extract features
    features = []
    labels = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            # Move to device
            images = images.to(device)
            
            # Extract features
            feature = model.extract_features(images)
            
            # Save features
            features.append(feature.cpu().numpy())
            labels.append(targets.numpy())
    
    # Concatenate features
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    return features, labels

def visualize_features(features, labels, method='tsne', n_components=2, save_path=None):
    """
    Visualize features
    
    Args:
        features: Extracted features
        labels: Ground truth labels
        method: Visualization method ('tsne' or 'pca')
        n_components: Number of components
        save_path: Path to save visualization
        
    Returns:
        Figure
    """
    # Reduce dimensionality
    if method == 'tsne':
        reduced_features = TSNE(n_components=n_components, random_state=42).fit_transform(features)
    elif method == 'pca':
        reduced_features = PCA(n_components=n_components, random_state=42).fit_transform(features)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    
    # Plot features
    if n_components == 2:
        # 2D plot
        plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', alpha=0.5)
        plt.colorbar(label='Class')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
    elif n_components == 3:
        # 3D plot
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            reduced_features[:, 0],
            reduced_features[:, 1],
            reduced_features[:, 2],
            c=labels,
            cmap='viridis',
            alpha=0.5
        )
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        plt.colorbar(scatter, label='Class')
    else:
        raise ValueError(f"Unsupported number of components: {n_components}")
    
    # Set title
    plt.title(f'Feature Visualization ({method.upper()})')
    
    # Save figure
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig