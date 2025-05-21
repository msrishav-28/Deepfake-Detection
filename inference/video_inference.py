# inference/video_inference.py
import os
import torch
import numpy as np
import cv2
import yaml
from typing import Dict, List, Tuple, Union, Optional
from tqdm import tqdm

from .inference import DeepfakeDetector
from .ensemble_inference import EnsembleDetector


class VideoDetector:
    """Video detector for deepfake detection in videos"""
    
    def __init__(
        self,
        detector: Union[DeepfakeDetector, EnsembleDetector],
        batch_size: int = 16,
        sample_rate: int = 30,
        device: str = None
    ):
        """
        Initialize the video detector
        
        Args:
            detector: DeepfakeDetector or EnsembleDetector instance
            batch_size: Batch size for processing frames
            sample_rate: Process one frame every N frames
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.detector = detector
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    def process_video(
        self,
        video_path: str,
        output_path: str = None,
        show_progress: bool = True,
        max_frames: int = None,
        extract_frames: bool = False,
        frames_dir: str = None
    ) -> Dict:
        """
        Process video for deepfake detection
        
        Args:
            video_path: Path to video file
            output_path: Path to save annotated video (optional)
            show_progress: Whether to show progress bar
            max_frames: Maximum number of frames to process
            extract_frames: Whether to save extracted frames
            frames_dir: Directory to save extracted frames
            
        Returns:
            Dictionary with detection results:
            {
                'frames_processed': int,
                'frames_with_faces': int,
                'predictions': List[str],
                'probabilities': List[float],
                'confidences': List[float],
                'frame_indices': List[int],
                'summary': Dict
            }
        """
        # Check if video exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found at {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        # Check if video opened successfully
        if not cap.isOpened():
            raise RuntimeError("Error opening video file")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frames to sample
        frames_to_sample = total_frames // self.sample_rate
        if max_frames is not None:
            frames_to_sample = min(frames_to_sample, max_frames)
        
        # Initialize video writer if output path is provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        else:
            out = None
        
        # Create frames directory if extracting frames
        if extract_frames:
            if not frames_dir:
                frames_dir = 'extracted_frames'
            os.makedirs(frames_dir, exist_ok=True)
        
        # Initialize results
        results = {
            'frames_processed': 0,
            'frames_with_faces': 0,
            'predictions': [],
            'probabilities': [],
            'confidences': [],
            'frame_indices': []
        }
        
        # Process video frames
        frame_idx = 0
        frame_count = 0
        
        # Create progress bar if requested
        if show_progress:
            pbar = tqdm(total=frames_to_sample, desc="Processing video")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if frame_idx % self.sample_rate == 0:
                # Increment frame count
                frame_count += 1
                
                # Convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame
                try:
                    # Preprocess frame to detect faces
                    faces = self.detector.preprocess_image(frame_rgb, return_faces=True)
                    
                    if isinstance(faces, tuple):
                        face_tensors, face_images = faces
                        has_faces = len(face_images) > 0
                    else:
                        face_tensors = faces
                        has_faces = face_tensors.size(0) > 0
                    
                    if has_faces:
                        # Increment faces count
                        results['frames_with_faces'] += 1
                        
                        # Get prediction
                        result = self.detector.predict(face_tensors)
                        
                        # Handle single or multiple faces
                        if 'aggregate' in result:
                            # Multiple faces
                            pred = result['aggregate']['prediction']
                            prob = result['aggregate']['probability']
                            conf = result['aggregate']['confidence']
                            
                            # Add face-specific metadata
                            results.setdefault('face_results', []).append({
                                'frame_idx': frame_idx,
                                'predictions': result['prediction'],
                                'probabilities': result['probability'],
                                'confidences': result['confidence']
                            })
                        else:
                            # Single face
                            pred = result['prediction']
                            prob = result['probability']
                            conf = result['confidence']
                        
                        # Store results
                        results['predictions'].append(pred)
                        results['probabilities'].append(prob)
                        results['confidences'].append(conf)
                        results['frame_indices'].append(frame_idx)
                        
                        # Annotate frame if output is requested
                        if out or extract_frames:
                            # Add prediction text
                            label = f"{pred.upper()}: {prob:.2f}"
                            color = (0, 255, 0) if pred == 'real' else (255, 0, 0)
                            
                            # Add text to frame
                            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                            
                            # Add face rectangles if available
                            if isinstance(faces, tuple) and len(face_images) > 0:
                                for i, face_img in enumerate(face_images):
                                    # Calculate face position (approximate)
                                    h, w = face_img.shape[:2]
                                    
                                    # This is a simplification - in real use cases, you'd need to track 
                                    # the actual coordinates from the face detector
                                    x1 = max(0, frame.shape[1]//2 - w//2)
                                    y1 = max(0, frame.shape[0]//2 - h//2)
                                    x2 = min(frame.shape[1], x1 + w)
                                    y2 = min(frame.shape[0], y1 + h)
                                    
                                    # Draw rectangle
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    else:
                        # No faces detected
                        if out or extract_frames:
                            cv2.putText(frame, "NO FACE DETECTED", (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                except Exception as e:
                    print(f"Error processing frame {frame_idx}: {e}")
                    
                    # Add error text to frame
                    if out or extract_frames:
                        cv2.putText(frame, f"ERROR: {str(e)[:50]}", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Save frame if extracting
                if extract_frames:
                    frame_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.jpg")
                    cv2.imwrite(frame_path, frame)
                
                # Write frame if output is requested
                if out:
                    out.write(frame)
                
                # Update progress bar
                if show_progress:
                    pbar.update(1)
                
                # Check if we've processed enough frames
                if max_frames is not None and frame_count >= max_frames:
                    break
            
            # Increment frame index
            frame_idx += 1
        
        # Close progress bar
        if show_progress:
            pbar.close()
        
        # Release video objects
        cap.release()
        if out:
            out.release()
        
        # Update results
        results['frames_processed'] = frame_count
        
        # Calculate summary statistics
        if results['predictions']:
            real_count = results['predictions'].count('real')
            fake_count = results['predictions'].count('fake')
            total_count = len(results['predictions'])
            
            real_percent = real_count / total_count * 100
            fake_percent = fake_count / total_count * 100
            
            avg_prob = sum(results['probabilities']) / total_count
            avg_conf = sum(results['confidences']) / total_count
            
            # Determine overall verdict
            if fake_percent > 70:
                verdict = "FAKE"
                confidence = "high"
            elif fake_percent > 40:
                verdict = "FAKE"
                confidence = "moderate"
            elif fake_percent > 20:
                verdict = "SUSPICIOUS"
                confidence = "low"
            else:
                verdict = "REAL"
                confidence = "high" if real_percent > 80 else "moderate"
            
            results['summary'] = {
                'real_frames': real_count,
                'fake_frames': fake_count,
                'real_percent': real_percent,
                'fake_percent': fake_percent,
                'avg_probability': avg_prob,
                'avg_confidence': avg_conf,
                'verdict': verdict,
                'confidence': confidence
            }
        else:
            results['summary'] = {
                'verdict': "UNKNOWN",
                'confidence': "none",
                'message': "No faces detected in the sampled frames"
            }
        
        return results
    
    def create_analysis_video(
        self,
        video_path: str,
        output_path: str,
        temporal_window: int = 5,
        show_progress: bool = True
    ) -> None:
        """
        Create a video with deepfake analysis visualization
        
        Args:
            video_path: Path to video file
            output_path: Path to save analysis video
            temporal_window: Number of frames to use for temporal smoothing
            show_progress: Whether to show progress bar
        """
        # Process video to get detection results
        results = self.process_video(
            video_path=video_path,
            show_progress=show_progress
        )
        
        # Check if we have results
        if not results['predictions']:
            print("No predictions found. Cannot create analysis video.")
            return
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        # Check if video opened successfully
        if not cap.isOpened():
            raise RuntimeError("Error opening video file")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height + 200))  # Extra height for graph
        
        # Get frame indices and probabilities
        frame_indices = np.array(results['frame_indices'])
        probabilities = np.array(results['probabilities'])
        
        # Create temporal smoothing if requested
        if temporal_window > 1:
            # Apply moving average filter
            from scipy.ndimage import uniform_filter1d
            smoothed_probs = uniform_filter1d(probabilities, size=temporal_window, mode='nearest')
        else:
            smoothed_probs = probabilities
        
        # Process video frames
        frame_idx = 0
        result_idx = 0
        
        # Create progress bar if requested
        if show_progress:
            pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Creating analysis video")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Check if this frame has a prediction
            if result_idx < len(frame_indices) and frame_idx == frame_indices[result_idx]:
                # Get prediction
                pred = results['predictions'][result_idx]
                prob = results['probabilities'][result_idx]
                conf = results['confidences'][result_idx]
                smoothed_prob = smoothed_probs[result_idx]
                
                # Annotate frame
                label = f"{pred.upper()}: {prob:.2f} (smoothed: {smoothed_prob:.2f})"
                color = (0, 255, 0) if pred == 'real' else (255, 0, 0)
                
                # Add text to frame
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # Create visualization frame
                viz_frame = np.ones((200, width, 3), dtype=np.uint8) * 255
                
                # Draw threshold line
                cv2.line(viz_frame, (0, 100), (width, 100), (0, 0, 0), 1)
                
                # Draw probability points and line
                for i in range(min(result_idx + 1, len(frame_indices))):
                    idx = frame_indices[i]
                    raw_prob = results['probabilities'][i]
                    smooth_prob = smoothed_probs[i]
                    
                    # Calculate x position based on frame index
                    x_pos = int(idx / frame_idx * width) if frame_idx > 0 else 0
                    
                    # Draw raw probability point
                    y_pos_raw = int(200 - raw_prob * 200)
                    cv2.circle(viz_frame, (x_pos, y_pos_raw), 3, (255, 0, 0), -1)
                    
                    # Draw smoothed probability point
                    y_pos_smooth = int(200 - smooth_prob * 200)
                    cv2.circle(viz_frame, (x_pos, y_pos_smooth), 3, (0, 0, 255), -1)
                    
                    # Draw line to next point if available
                    if i < min(result_idx, len(frame_indices) - 1):
                        next_idx = frame_indices[i + 1]
                        next_raw_prob = results['probabilities'][i + 1]
                        next_smooth_prob = smoothed_probs[i + 1]
                        
                        next_x_pos = int(next_idx / frame_idx * width) if frame_idx > 0 else 0
                        next_y_pos_raw = int(200 - next_raw_prob * 200)
                        next_y_pos_smooth = int(200 - next_smooth_prob * 200)
                        
                        cv2.line(viz_frame, (x_pos, y_pos_raw), (next_x_pos, next_y_pos_raw), (255, 0, 0), 1)
                        cv2.line(viz_frame, (x_pos, y_pos_smooth), (next_x_pos, next_y_pos_smooth), (0, 0, 255), 2)
                
                # Add legend
                cv2.putText(viz_frame, "Raw Probability", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(viz_frame, "Smoothed Probability", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(viz_frame, "Threshold (0.5)", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Combine frame and visualization
                combined_frame = np.vstack((frame, viz_frame))
                
                # Write combined frame
                out.write(combined_frame)
                
                # Increment result index
                result_idx += 1
            else:
                # No prediction for this frame, just write the original frame with visualization
                viz_frame = np.ones((200, width, 3), dtype=np.uint8) * 255
                
                # Draw threshold line
                cv2.line(viz_frame, (0, 100), (width, 100), (0, 0, 0), 1)
                
                # Draw probability points and line from previous results
                for i in range(min(result_idx, len(frame_indices))):
                    idx = frame_indices[i]
                    raw_prob = results['probabilities'][i]
                    smooth_prob = smoothed_probs[i]
                    
                    # Calculate x position based on frame index
                    x_pos = int(idx / frame_idx * width) if frame_idx > 0 else 0
                    
                    # Draw raw probability point
                    y_pos_raw = int(200 - raw_prob * 200)
                    cv2.circle(viz_frame, (x_pos, y_pos_raw), 3, (255, 0, 0), -1)
                    
                    # Draw smoothed probability point
                    y_pos_smooth = int(200 - smooth_prob * 200)
                    cv2.circle(viz_frame, (x_pos, y_pos_smooth), 3, (0, 0, 255), -1)
                    
                    # Draw line to next point if available
                    if i < min(result_idx - 1, len(frame_indices) - 1):
                        next_idx = frame_indices[i + 1]
                        next_raw_prob = results['probabilities'][i + 1]
                        next_smooth_prob = smoothed_probs[i + 1]
                        
                        next_x_pos = int(next_idx / frame_idx * width) if frame_idx > 0 else 0
                        next_y_pos_raw = int(200 - next_raw_prob * 200)
                        next_y_pos_smooth = int(200 - next_smooth_prob * 200)
                        
                        cv2.line(viz_frame, (x_pos, y_pos_raw), (next_x_pos, next_y_pos_raw), (255, 0, 0), 1)
                        cv2.line(viz_frame, (x_pos, y_pos_smooth), (next_x_pos, next_y_pos_smooth), (0, 0, 255), 2)
                
                # Add legend
                cv2.putText(viz_frame, "Raw Probability", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(viz_frame, "Smoothed Probability", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(viz_frame, "Threshold (0.5)", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Combine frame and visualization
                combined_frame = np.vstack((frame, viz_frame))
                
                # Write combined frame
                out.write(combined_frame)
            
            # Increment frame index
            frame_idx += 1
            
            # Update progress bar
            if show_progress:
                pbar.update(1)
        
        # Close progress bar
        if show_progress:
            pbar.close()
        
        # Release video objects
        cap.release()
        out.release()
        
        print(f"Analysis video saved to {output_path}")


if __name__ == "__main__":
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Deepfake Detection Video Processing")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--output", type=str, help="Path to save annotated video")
    parser.add_argument("--analysis", type=str, help="Path to save analysis video")
    parser.add_argument("--ensemble", action="store_true", help="Use ensemble detection")
    parser.add_argument("--sample_rate", type=int, default=30, help="Process one frame every N frames")
    parser.add_argument("--max_frames", type=int, help="Maximum number of frames to process")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing frames")
    parser.add_argument("--extract_frames", action="store_true", help="Extract processed frames")
    parser.add_argument("--frames_dir", type=str, help="Directory to save extracted frames")
    args = parser.parse_args()
    
    # Load detector
    if args.ensemble:
        from ensemble_inference import EnsembleDetector
        detector = EnsembleDetector(config_path=args.config)
    else:
        from inference import load_detector
        detector = load_detector(args.config)
    
    # Create video detector
    video_detector = VideoDetector(
        detector=detector,
        batch_size=args.batch_size,
        sample_rate=args.sample_rate
    )
    
    # Process video
    results = video_detector.process_video(
        video_path=args.video,
        output_path=args.output,
        max_frames=args.max_frames,
        extract_frames=args.extract_frames,
        frames_dir=args.frames_dir
    )
    
    # Print summary
    print("\nVIDEO ANALYSIS SUMMARY:")
    print(f"Processed {results['frames_processed']} frames")
    print(f"Frames with faces: {results['frames_with_faces']}")
    
    if 'summary' in results:
        summary = results['summary']
        
        if 'real_frames' in summary:
            print(f"Real frames: {summary['real_frames']} ({summary['real_percent']:.1f}%)")
            print(f"Fake frames: {summary['fake_frames']} ({summary['fake_percent']:.1f}%)")
            print(f"Average probability of being fake: {summary['avg_probability']:.4f}")
            print(f"Average confidence: {summary['avg_confidence']:.4f}")
        
        print(f"\nOverall verdict: Video is likely {summary['verdict']} with {summary['confidence']} confidence")
        
        if 'message' in summary:
            print(f"Message: {summary['message']}")
    
    # Create analysis video if requested
    if args.analysis:
        video_detector.create_analysis_video(
            video_path=args.video,
            output_path=args.analysis,
            temporal_window=5
        )