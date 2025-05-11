# data/preprocessing/frame_extraction.py
import os
import cv2
import numpy as np
from tqdm import tqdm

def extract_frames(video_path, output_dir, sample_rate=30, max_frames=100):
    """
    Extract frames from a video
    
    Args:
        video_path: Path to video file
        output_dir: Output directory
        sample_rate: Sample one frame every N frames
        max_frames: Maximum number of frames to extract
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_count = min(frame_count, max_frames)
    
    # Sample frames at regular intervals
    sample_indices = np.linspace(0, frame_count - 1, sample_count, dtype=int)
    
    # Process frames
    frame_idx = 0
    saved_count = 0
    
    while cap.isOpened():
        # Get next frame
        ret, frame = cap.read()
        
        if not ret:
            break
            
        # Process sampled frames
        if frame_idx in sample_indices:
            # Save frame
            frame_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.png")
            cv2.imwrite(frame_path, frame)
            saved_count += 1
                
        frame_idx += 1
        
        # Stop if we've processed enough frames
        if frame_idx >= frame_count:
            break
            
    # Release video
    cap.release()
    
    return saved_count