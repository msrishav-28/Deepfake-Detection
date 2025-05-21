# inference/ensemble_inference.py
import os
import torch
import numpy as np
import cv2
import yaml
from typing import Dict, List, Tuple, Union, Optional

from .inference import DeepfakeDetector, load_detector


class EnsembleDetector:
    """Ensemble detector for combining multiple deepfake detection models"""
    
    def __init__(
        self,
        detectors: List[DeepfakeDetector] = None,
        ensemble_method: str = "average",
        weights: List[float] = None,
        config_path: str = None
    ):
        """
        Initialize the ensemble detector
        
        Args:
            detectors: List of DeepfakeDetector instances
            ensemble_method: Method for combining predictions ("average", "voting", "weighted", "max")
            weights: Weights for weighted average (if ensemble_method="weighted")
            config_path: Path to ensemble configuration file (alternative to detectors)
        """
        # Initialize from config if provided
        if config_path:
            self._init_from_config(config_path)
        else:
            self.detectors = detectors or []
            self.ensemble_method = ensemble_method
            
            # Check weights if using weighted average
            if ensemble_method == "weighted":
                if not weights or len(weights) != len(detectors):
                    raise ValueError("Weights must be provided for each detector when using weighted average")
                self.weights = weights
            else:
                self.weights = weights or [1.0] * len(self.detectors)
        
        # Normalize weights
        if self.weights:
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]
    
    def _init_from_config(self, config_path: str):
        """
        Initialize from configuration file
        
        Args:
            config_path: Path to configuration file
        """
        # Check if config file exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check if ensemble section exists
        if 'ensemble' not in config:
            raise ValueError("Invalid config file: 'ensemble' section missing")
        
        # Extract parameters
        ensemble_config = config['ensemble']
        self.ensemble_method = ensemble_config.get('method', 'average')
        self.weights = ensemble_config.get('weights', None)
        
        # Load detectors
        models = ensemble_config.get('models', [])
        self.detectors = []
        
        for model_config in models:
            # Extract model parameters
            model_type = model_config.get('type')
            model_path = model_config.get('checkpoint')
            
            # Check if required parameters exist
            if not model_type or not model_path:
                raise ValueError("Invalid model configuration: type and checkpoint required")
            
            # Create detector
            detector = DeepfakeDetector(
                model_type=model_type,
                model_path=model_path,
                device=config.get('device', None),
                face_detector=config.get('face_detector', True),
                config={'model': model_config.get('params', {})}
            )
            
            self.detectors.append(detector)
        
        # Check weights if using weighted average
        if self.ensemble_method == "weighted" and (not self.weights or len(self.weights) != len(self.detectors)):
            raise ValueError("Weights must be provided for each detector when using weighted average")
        
        # Default weights if not provided
        if not self.weights:
            self.weights = [1.0] * len(self.detectors)
        
        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
    
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
        # Use the first detector for preprocessing
        if not self.detectors:
            raise ValueError("No detectors available in ensemble")
        
        return self.detectors[0].preprocess_image(image, return_faces=return_faces)
    
    def predict(
        self, 
        image: Union[str, np.ndarray, torch.Tensor],
        return_features: bool = False
    ) -> Dict:
        """
        Predict if an image is real or fake using ensemble
        
        Args:
            image: Path to image, numpy array, or preprocessed tensor
            return_features: Whether to return the extracted features
            
        Returns:
            Dictionary with prediction results:
            {
                'probability': float,    # Ensemble probability of being fake (0-1)
                'prediction': str,       # 'real' or 'fake'
                'confidence': float,     # Confidence of prediction (0-1)
                'faces': int,            # Number of faces detected
                'models': List[Dict],    # Individual model predictions
                'features': List[tensor] # Feature vectors (if return_features=True)
            }
        """
        # Check if detectors exist
        if not self.detectors:
            raise ValueError("No detectors available in ensemble")
        
        # Process image to get faces
        if not isinstance(image, torch.Tensor):
            tensor, faces = self.preprocess_image(image, return_faces=True)
            n_faces = len(faces)
        else:
            tensor = image
            n_faces = tensor.size(0) if tensor.dim() == 4 else 1
            faces = None
        
        # Get predictions from each detector
        model_results = []
        all_features = []
        
        for i, detector in enumerate(self.detectors):
            # Get prediction
            result = detector.predict(tensor, return_features=return_features)
            
            # Store result
            model_results.append(result)
            
            # Store features if requested
            if return_features and 'features' in result:
                all_features.append(result['features'])
        
        # Combine predictions based on ensemble method
        if n_faces == 1:
            # Single face
            if self.ensemble_method == "voting":
                # Simple majority voting
                votes = [1 if r['prediction'] == 'fake' else 0 for r in model_results]
                fake_votes = sum(votes)
                real_votes = len(votes) - fake_votes
                ensemble_prob = fake_votes / len(votes)
                ensemble_pred = 'fake' if fake_votes > real_votes else 'real'
                ensemble_conf = max(fake_votes, real_votes) / len(votes)
            
            elif self.ensemble_method == "weighted":
                # Weighted average
                probs = [r['probability'] for r in model_results]
                ensemble_prob = sum(p * w for p, w in zip(probs, self.weights))
                ensemble_pred = 'fake' if ensemble_prob > 0.5 else 'real'
                ensemble_conf = ensemble_prob if ensemble_pred == 'fake' else 1 - ensemble_prob
            
            elif self.ensemble_method == "max":
                # Maximum confidence
                confs = [r['confidence'] for r in model_results]
                max_idx = np.argmax(confs)
                ensemble_prob = model_results[max_idx]['probability']
                ensemble_pred = model_results[max_idx]['prediction']
                ensemble_conf = model_results[max_idx]['confidence']
            
            else:  # default: average
                # Simple average
                probs = [r['probability'] for r in model_results]
                ensemble_prob = sum(probs) / len(probs)
                ensemble_pred = 'fake' if ensemble_prob > 0.5 else 'real'
                ensemble_conf = ensemble_prob if ensemble_pred == 'fake' else 1 - ensemble_prob
            
            # Prepare result
            result = {
                'probability': float(ensemble_prob),
                'prediction': ensemble_pred,
                'confidence': float(ensemble_conf),
                'faces': n_faces,
                'models': model_results
            }
            
            # Add features if requested
            if return_features:
                result['features'] = all_features
        
        else:
            # Multiple faces - apply ensemble to each face
            ensemble_probs = []
            ensemble_preds = []
            ensemble_confs = []
            
            for face_idx in range(n_faces):
                if self.ensemble_method == "voting":
                    # Simple majority voting
                    votes = [1 if r['prediction'][face_idx] == 'fake' else 0 for r in model_results]
                    fake_votes = sum(votes)
                    real_votes = len(votes) - fake_votes
                    face_prob = fake_votes / len(votes)
                    face_pred = 'fake' if fake_votes > real_votes else 'real'
                    face_conf = max(fake_votes, real_votes) / len(votes)
                
                elif self.ensemble_method == "weighted":
                    # Weighted average
                    probs = [r['probability'][face_idx] for r in model_results]
                    face_prob = sum(p * w for p, w in zip(probs, self.weights))
                    face_pred = 'fake' if face_prob > 0.5 else 'real'
                    face_conf = face_prob if face_pred == 'fake' else 1 - face_prob
                
                elif self.ensemble_method == "max":
                    # Maximum confidence
                    confs = [r['confidence'][face_idx] for r in model_results]
                    max_idx = np.argmax(confs)
                    face_prob = model_results[max_idx]['probability'][face_idx]
                    face_pred = model_results[max_idx]['prediction'][face_idx]
                    face_conf = model_results[max_idx]['confidence'][face_idx]
                
                else:  # default: average
                    # Simple average
                    probs = [r['probability'][face_idx] for r in model_results]
                    face_prob = sum(probs) / len(probs)
                    face_pred = 'fake' if face_prob > 0.5 else 'real'
                    face_conf = face_prob if face_pred == 'fake' else 1 - face_prob
                
                ensemble_probs.append(face_prob)
                ensemble_preds.append(face_pred)
                ensemble_confs.append(face_conf)
            
            # Prepare result
            result = {
                'probability': ensemble_probs,
                'prediction': ensemble_preds,
                'confidence': ensemble_confs,
                'faces': n_faces,
                'models': model_results
            }
            
            # Add features if requested
            if return_features:
                result['features'] = all_features
            
            # Add aggregate result
            result['aggregate'] = {
                'probability': float(np.mean(ensemble_probs)),
                'prediction': 'fake' if np.mean(ensemble_probs) > 0.5 else 'real',
                'confidence': float(np.mean(ensemble_confs))
            }
        
        return result
    
    def explain(
        self, 
        image: Union[str, np.ndarray],
        model_index: int = None,
        face_index: int = 0
    ) -> Dict:
        """
        Generate explanation for the ensemble's prediction
        
        Args:
            image: Path to image or numpy array
            model_index: Index of the model to explain (None for all models)
            face_index: Index of the face to explain (if multiple faces detected)
            
        Returns:
            Dictionary with explanation visualizations for each model
        """
        # Check if detectors exist
        if not self.detectors:
            raise ValueError("No detectors available in ensemble")
        
        # Preprocess image
        tensor, faces = self.preprocess_image(image, return_faces=True)
        
        # Check if face index is valid
        if face_index >= len(faces):
            raise ValueError(f"Face index {face_index} is out of range. Only {len(faces)} faces detected.")
        
        # Get predictions from ensemble
        ensemble_result = self.predict(tensor)
        
        # Initialize explanation dict
        explanation = {
            'original': faces[face_index],
            'ensemble_prediction': ensemble_result,
            'models': {}
        }
        
        # Generate explanations for specific model or all models
        if model_index is not None:
            # Check if model index is valid
            if model_index >= len(self.detectors):
                raise ValueError(f"Model index {model_index} is out of range. Only {len(self.detectors)} models available.")
            
            # Get explanation for specific model
            detector = self.detectors[model_index]
            model_explanation = detector.explain(image, face_index=face_index)
            
            explanation['models'][f"model_{model_index}"] = model_explanation
        else:
            # Get explanation for all models
            for i, detector in enumerate(self.detectors):
                model_explanation = detector.explain(image, face_index=face_index)
                explanation['models'][f"model_{i}"] = model_explanation
        
        return explanation


if __name__ == "__main__":
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Deepfake Detection Ensemble Inference")
    parser.add_argument("--config", type=str, required=True, help="Path to ensemble configuration file")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--explain", action="store_true", help="Generate explanation")
    parser.add_argument("--model_index", type=int, help="Index of the model to explain")
    parser.add_argument("--output", type=str, help="Path to save explanation visualization")
    args = parser.parse_args()
    
    # Load ensemble detector
    ensemble = EnsembleDetector(config_path=args.config)
    
    # Run prediction
    result = ensemble.predict(args.image)
    
    # Print result
    if 'aggregate' in result:
        # Multiple faces
        print(f"Detected {result['faces']} faces")
        print(f"Aggregate prediction: {result['aggregate']['prediction']}")
        print(f"Aggregate probability of being fake: {result['aggregate']['probability']:.4f}")
        print(f"Aggregate confidence: {result['aggregate']['confidence']:.4f}")
        
        print("\nPer-face predictions:")
        for i, (pred, prob, conf) in enumerate(zip(result['prediction'], result['probability'], result['confidence'])):
            print(f"Face {i+1}: {pred} (Prob: {prob:.4f}, Conf: {conf:.4f})")
    else:
        # Single face
        print(f"Ensemble prediction: {result['prediction']}")
        print(f"Ensemble probability of being fake: {result['probability']:.4f}")
        print(f"Ensemble confidence: {result['confidence']:.4f}")
        
        print("\nIndividual model predictions:")
        for i, model_result in enumerate(result['models']):
            print(f"Model {i+1}: {model_result['prediction']} (Prob: {model_result['probability']:.4f}, Conf: {model_result['confidence']:.4f})")
    
    # Generate explanation if requested
    if args.explain:
        explanation = ensemble.explain(args.image, model_index=args.model_index)
        
        # Visualize explanation
        import matplotlib.pyplot as plt
        
        # Determine how many models to visualize
        if args.model_index is not None:
            models_to_plot = [f"model_{args.model_index}"]
        else:
            models_to_plot = list(explanation['models'].keys())
        
        # Create a figure for each model
        for model_key in models_to_plot:
            model_exp = explanation['models'][model_key]
            
            plt.figure(figsize=(15, 5))
            plt.suptitle(f"Explanation for {model_key}")
            
            # Original image
            plt.subplot(1, 3, 1)
            plt.imshow(model_exp['original'])
            plt.title(f"Original - {model_exp['prediction']['prediction'].upper()}")
            plt.axis('off')
            
            # Grad-CAM
            if 'grad_cam' in model_exp:
                plt.subplot(1, 3, 2)
                plt.imshow(model_exp['grad_cam']['overlay'])
                plt.title('Grad-CAM')
                plt.axis('off')
            
            # Attention map
            if 'attention' in model_exp:
                plt.subplot(1, 3, 3)
                plt.imshow(model_exp['attention']['overlay'])
                plt.title('Attention Map')
                plt.axis('off')
            
            plt.tight_layout()
            
            # Save or show
            if args.output:
                # Create model-specific output path
                base, ext = os.path.splitext(args.output)
                model_output = f"{base}_{model_key}{ext}"
                plt.savefig(model_output)
                print(f"Explanation for {model_key} saved to {model_output}")
            else:
                plt.show()