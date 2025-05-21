# inference/inference.py
import os
import torch
import numpy as np
import cv2
import yaml
from PIL import Image
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Optional

from data.preprocessing.face_extraction import setup_face_detector, extract_faces
from data.preprocessing.normalization import normalize_face
from models.model_zoo.model_factory import create_model
from evaluation.visualization.grad_cam import visualize_grad_cam
from evaluation.visualization.attention_maps import visualize_attention_maps


class DeepfakeDetector:
    """Base class for deepfake detection inference"""
    
    def __init__(
        self,
        model_type: str,
        model_path: str,
        device: str = None,
        face_detector: bool = True,
        config: Dict = None
    ):
        """
        Initialize the deepfake detector
        
        Args:
            model_type: Type of model ('vit', 'deit', 'swin')
            model_path: Path to model checkpoint
            device: Device to run inference on ('cpu' or 'cuda')
            face_detector: Whether to use face detection
            config: Model configuration
        """
        # Set device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Save parameters
        self.model_type = model_type
        self.model_path = model_path
        self.config = config or {}
        
        # Load model
        self.model = self._load_model()
        
        # Setup face detector if needed
        if face_detector:
            try:
                self.face_detector = setup_face_detector(device='cpu')
            except Exception as e:
                print(f"Warning: Could not initialize face detector: {e}")
                print("Face detection will be disabled.")
                self.face_detector = None
        else:
            self.face_detector = None
    
    def _load_model(self) -> torch.nn.Module:
        """
        Load model from checkpoint
        
        Returns:
            PyTorch model
        """
        # Create model based on type
        if self.model_type in ['vit', 'deit', 'swin']:
            # Use model factory for standard models
            model = create_model(self.model_type, **self.config.get('model', {}))
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Check if checkpoint exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {self.model_path}")
        
        # Load checkpoint
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            print(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model checkpoint: {e}")
        
        # Move model to device and set to evaluation mode
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def preprocess_image(
        self, 
        image: Union[str, np.ndarray],
        return_faces: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[np.ndarray]]]:
        """
        Preprocess an image for inference
        
        Args:
            image: Path to image or numpy array
            return_faces: Whether to return the extracted faces
            
        Returns:
            If return_faces=False: 
                Preprocessed tensor of shape (B, C, H, W)
            If return_faces=True: 
                Tuple of (tensor, face_images)
        """
        # Load image if path is provided
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image not found at {image}")
            
            # Load image
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # Assume numpy array
            img = image
            
            # Convert to RGB if needed
            if len(img.shape) == 3 and img.shape[2] == 3:
                # Check if image is in BGR format (from OpenCV)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Extract faces if detector is available
        if self.face_detector is not None:
            faces = extract_faces(img, self.face_detector)
            
            if not faces:
                print("Warning: No faces detected in the image. Using the whole image.")
                faces = [img]
        else:
            # Use whole image if no face detector
            faces = [img]
        
        # Process each face
        tensors = []
        for face in faces:
            # Normalize face
            face_tensor = normalize_face(face)
            tensors.append(face_tensor)
        
        # Stack tensors
        batch_tensor = torch.stack(tensors)
        
        if return_faces:
            return batch_tensor, faces
        else:
            return batch_tensor
    
    def predict(
        self, 
        image: Union[str, np.ndarray, torch.Tensor],
        return_features: bool = False
    ) -> Dict:
        """
        Predict if an image is real or fake
        
        Args:
            image: Path to image, numpy array, or preprocessed tensor
            return_features: Whether to return the extracted features
            
        Returns:
            Dictionary with prediction results:
            {
                'probability': float,  # Probability of being fake (0-1)
                'prediction': str,     # 'real' or 'fake'
                'confidence': float,   # Confidence of prediction (0-1)
                'faces': int,          # Number of faces detected
                'features': tensor,    # Feature vectors (if return_features=True)
            }
        """
        # Process image if it's not already a tensor
        if not isinstance(image, torch.Tensor):
            tensor, faces = self.preprocess_image(image, return_faces=True)
            n_faces = len(faces)
        else:
            tensor = image
            n_faces = tensor.size(0) if tensor.dim() == 4 else 1
            # Ensure proper dimensions
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            faces = None
        
        # Move tensor to device
        tensor = tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            # Extract features if requested
            if return_features:
                if hasattr(self.model, 'extract_features'):
                    features = self.model.extract_features(tensor)
                else:
                    # Default feature extraction for standard models
                    if self.model_type in ['vit', 'deit']:
                        # For ViT/DeiT, use the output of forward_features
                        features = self.model.forward_features(tensor)
                        if isinstance(features, tuple):
                            features = features[0]  # For DeiT, take the class token output
                    elif self.model_type == 'swin':
                        # For Swin, use the output of the last stage
                        features = self.model.forward_features(tensor)
                    else:
                        # Generic feature extraction
                        features = tensor
                        for name, module in self.model.named_children():
                            if name == 'head' or name == 'fc':
                                break
                            features = module(features)
                
                # Get final predictions
                outputs = self.model(tensor)
            else:
                features = None
                outputs = self.model(tensor)
            
            # Convert outputs to probabilities
            if outputs.shape[-1] > 1:
                # Multi-class output
                probs = F.softmax(outputs, dim=1)
                fake_probs = probs[:, 1]  # Assuming fake is class 1
            else:
                # Binary output
                fake_probs = torch.sigmoid(outputs).squeeze()
            
            # Get predictions
            if fake_probs.dim() == 0:
                fake_probs = fake_probs.unsqueeze(0)
            
            preds = (fake_probs > 0.5).long()
            
            # Calculate confidence
            confidence = torch.where(preds == 1, fake_probs, 1 - fake_probs)
            
            # Convert to numpy
            fake_probs = fake_probs.cpu().numpy()
            preds = preds.cpu().numpy()
            confidence = confidence.cpu().numpy()
            
            # Prepare results
            if n_faces == 1:
                result = {
                    'probability': float(fake_probs[0]),
                    'prediction': 'fake' if preds[0] == 1 else 'real',
                    'confidence': float(confidence[0]),
                    'faces': n_faces
                }
                if return_features:
                    result['features'] = features.cpu()
            else:
                # Multiple faces
                result = {
                    'probability': fake_probs.tolist(),
                    'prediction': ['fake' if p == 1 else 'real' for p in preds],
                    'confidence': confidence.tolist(),
                    'faces': n_faces
                }
                if return_features:
                    result['features'] = features.cpu()
                
                # Add aggregate result
                result['aggregate'] = {
                    'probability': float(np.mean(fake_probs)),
                    'prediction': 'fake' if np.mean(fake_probs) > 0.5 else 'real',
                    'confidence': float(np.mean(confidence))
                }
        
        return result
    
    def explain(self, image: Union[str, np.ndarray], face_index: int = 0) -> Dict:
        """
        Generate explanation for the model's prediction
        
        Args:
            image: Path to image or numpy array
            face_index: Index of the face to explain (if multiple faces detected)
            
        Returns:
            Dictionary with explanation visualizations:
            {
                'original': original face image,
                'prediction': prediction result,
                'grad_cam': Grad-CAM heatmap,
                'attention': attention map (for transformer models)
            }
        """
        # Preprocess image
        tensor, faces = self.preprocess_image(image, return_faces=True)
        
        # Check if face index is valid
        if face_index >= len(faces):
            raise ValueError(f"Face index {face_index} is out of range. Only {len(faces)} faces detected.")
        
        # Get specific face and tensor
        face = faces[face_index]
        face_tensor = tensor[face_index].unsqueeze(0).to(self.device)
        
        # Get prediction
        result = self.predict(face_tensor)
        
        # Initialize explanation dict
        explanation = {
            'original': face,
            'prediction': result
        }
        
        # Generate Grad-CAM visualization
        try:
            # Find target layer based on model type
            if self.model_type == 'vit':
                target_layer = self.model.blocks[-1]
            elif self.model_type == 'deit':
                target_layer = self.model.blocks[-1]
            elif self.model_type == 'swin':
                target_layer = self.model.layers[-1]
            else:
                raise ValueError(f"Grad-CAM not implemented for model type: {self.model_type}")
            
            # Create GradCAM
            from evaluation.visualization.grad_cam import GradCAM
            grad_cam = GradCAM(self.model, target_layer)
            
            # Generate heatmap
            cam = grad_cam(face_tensor)
            
            # Convert to heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Resize to match image size
            heatmap = cv2.resize(heatmap, (face.shape[1], face.shape[0]))
            
            # Overlay heatmap on image
            superimposed = heatmap * 0.4 + face * 0.6
            superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
            
            explanation['grad_cam'] = {
                'heatmap': heatmap,
                'overlay': superimposed
            }
        except Exception as e:
            print(f"Warning: Failed to generate Grad-CAM: {e}")
        
        # Generate attention map visualization (for transformer models)
        if self.model_type in ['vit', 'deit']:
            try:
                # Get attention maps
                from evaluation.visualization.attention_maps import get_attention_maps
                attention_maps = get_attention_maps(self.model, face_tensor, head_idx=0)
                
                if attention_maps:
                    # Get attention map from the last layer
                    attn_map = attention_maps[-1]
                    
                    # Reshape to square grid
                    size = int(np.sqrt(attn_map.shape[0]))
                    attn_grid = attn_map.reshape(size, size).cpu().numpy()
                    
                    # Resize to match image size
                    attn_resized = cv2.resize(attn_grid, (face.shape[1], face.shape[0]))
                    
                    # Create heatmap
                    attn_heatmap = cv2.applyColorMap(np.uint8(255 * attn_resized), cv2.COLORMAP_VIRIDIS)
                    attn_heatmap = cv2.cvtColor(attn_heatmap, cv2.COLOR_BGR2RGB)
                    
                    # Overlay on image
                    attn_overlay = attn_heatmap * 0.4 + face * 0.6
                    attn_overlay = np.clip(attn_overlay, 0, 255).astype(np.uint8)
                    
                    explanation['attention'] = {
                        'map': attn_resized,
                        'heatmap': attn_heatmap,
                        'overlay': attn_overlay
                    }
            except Exception as e:
                print(f"Warning: Failed to generate attention map: {e}")
        
        return explanation


def load_detector(config_path: str) -> DeepfakeDetector:
    """
    Load a deepfake detector from a configuration file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        DeepfakeDetector instance
    """
    # Check if config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check if model section exists
    if 'model' not in config:
        raise ValueError("Invalid config file: 'model' section missing")
    
    # Extract parameters
    model_type = config['model'].get('type')
    model_path = config['model'].get('checkpoint')
    
    # Check if required parameters exist
    if not model_type:
        raise ValueError("Invalid config file: model type not specified")
    if not model_path:
        raise ValueError("Invalid config file: model checkpoint not specified")
    
    # Initialize detector
    detector = DeepfakeDetector(
        model_type=model_type,
        model_path=model_path,
        device=config.get('device', None),
        face_detector=config.get('face_detector', True),
        config=config
    )
    
    return detector


if __name__ == "__main__":
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Deepfake Detection Inference")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--explain", action="store_true", help="Generate explanation")
    parser.add_argument("--output", type=str, help="Path to save explanation visualization")
    args = parser.parse_args()
    
    # Load detector
    detector = load_detector(args.config)
    
    # Run prediction
    result = detector.predict(args.image)
    
    # Print result
    print(f"Prediction: {result['prediction']}")
    print(f"Probability of being fake: {result['probability']:.4f}")
    print(f"Confidence: {result['confidence']:.4f}")
    
    # Generate explanation if requested
    if args.explain:
        explanation = detector.explain(args.image)
        
        # Visualize explanation
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(explanation['original'])
        plt.title(f"Original - {result['prediction'].upper()}")
        plt.axis('off')
        
        # Grad-CAM
        if 'grad_cam' in explanation:
            plt.subplot(1, 3, 2)
            plt.imshow(explanation['grad_cam']['overlay'])
            plt.title('Grad-CAM')
            plt.axis('off')
        
        # Attention map
        if 'attention' in explanation:
            plt.subplot(1, 3, 3)
            plt.imshow(explanation['attention']['overlay'])
            plt.title('Attention Map')
            plt.axis('off')
        
        plt.tight_layout()
        
        # Save or show
        if args.output:
            plt.savefig(args.output)
            print(f"Explanation saved to {args.output}")
        else:
            plt.show()