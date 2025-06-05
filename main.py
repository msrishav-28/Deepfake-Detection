#!/usr/bin/env python3
"""
Main entry point for Deepfake Detection System
"""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(
        description="Deepfake Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model
  python main.py train --config configs/training_config.yaml
  
  # Evaluate a model
  python main.py evaluate --config configs/evaluation_config.yaml
  
  # Run inference on an image
  python main.py detect --image path/to/image.jpg --config configs/inference_config.yaml
  
  # Run inference on a video
  python main.py detect --video path/to/video.mp4 --config configs/video_inference_config.yaml
  
  # Start API server
  python main.py serve --config configs/deployment_config.yaml
  
  # Preprocess dataset
  python main.py preprocess --dataset celebdf --input /path/to/raw --output /path/to/processed
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train a deepfake detection model')
    train_parser.add_argument('--config', type=str, required=True, help='Path to training config')
    train_parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--config', type=str, required=True, help='Path to evaluation config')
    
    # Detection command
    detect_parser = subparsers.add_parser('detect', help='Run deepfake detection')
    detect_parser.add_argument('--config', type=str, required=True, help='Path to inference config')
    detect_parser.add_argument('--image', type=str, help='Path to image file')
    detect_parser.add_argument('--video', type=str, help='Path to video file')
    detect_parser.add_argument('--output', type=str, help='Output directory')
    
    # API server command
    serve_parser = subparsers.add_parser('serve', help='Start API server')
    serve_parser.add_argument('--config', type=str, required=True, help='Path to deployment config')
    serve_parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind')
    serve_parser.add_argument('--port', type=int, default=8000, help='Port to bind')
    
    # Preprocessing command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess dataset')
    preprocess_parser.add_argument('--dataset', type=str, required=True, 
                                 choices=['celebdf', 'faceforensics'], 
                                 help='Dataset type')
    preprocess_parser.add_argument('--input', type=str, required=True, help='Input directory')
    preprocess_parser.add_argument('--output', type=str, required=True, help='Output directory')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    # Execute commands
    if args.command == 'train':
        from training.train import train_model
        train_model(args.config, os.path.dirname(args.config))
        
    elif args.command == 'evaluate':
        from evaluation.cross_dataset.cross_evaluation import evaluate_cross_datasets
        evaluate_cross_datasets(args.config, os.path.dirname(args.config))
        
    elif args.command == 'detect':
        if args.image:
            from inference.inference import load_detector
            detector = load_detector(args.config)
            result = detector.predict(args.image)
            print(f"Result: {result}")
        elif args.video:
            from inference.video_inference import VideoDetector
            from inference.inference import load_detector
            detector = load_detector(args.config)
            video_detector = VideoDetector(detector)
            results = video_detector.process_video(args.video, args.output)
            print(f"Results: {results['summary']}")
        else:
            print("Error: Either --image or --video must be specified")
            return 1
            
    elif args.command == 'serve':
        from inference.deployment.api import create_api, run_server
        app = create_api(args.config)
        run_server(app, host=args.host, port=args.port)
        
    elif args.command == 'preprocess':
        if args.dataset == 'celebdf':
            from scripts.preprocess_celebdf import process_videos
            # Implementation needed
        elif args.dataset == 'faceforensics':
            from scripts.preprocess_faceforensics import main as preprocess_ff
            # Implementation needed
    
    return 0

if __name__ == "__main__":
    sys.exit(main())