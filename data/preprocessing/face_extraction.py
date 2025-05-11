# data/preprocessing/face_extraction.py
import os
import cv2
import numpy as np
from tqdm import tqdm
from facenet_pytorch import MTCNN

def extract_faces(image, detector, margin=0.2, min_size=40):
    """
    Extract faces from an image using MTCNN
    
    Args:
        image: Input image
        detector: MTCNN detector
        margin: Margin to add around the face
        min_size: Minimum face size to detect
        
    Returns:
        List of extracted face images
    """
    # Convert to RGB if needed
    if image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Detect faces
    boxes, probs = detector.detect(image, landmarks=False)
    
    # If no faces, return empty list
    if boxes is None:
        return []
    
    # Extract face images
    face_images = []
    for box, prob in zip(boxes, probs):
        # Skip low confidence detections
        if prob < 0.9:
            continue
            
        # Get coordinates
        x1, y1, x2, y2 = [int(b) for b in box]
        
        # Add margin
        h = y2 - y1
        w = x2 - x1
        y1 = max(0, int(y1 - margin * h))
        y2 = min(image.shape[0], int(y2 + margin * h))
        x1 = max(0, int(x1 - margin * w))
        x2 = min(image.shape[1], int(x2 + margin * w))
        
        # Extract face
        face = image[y1:y2, x1:x2]
        
        # Skip small faces
        if face.shape[0] < min_size or face.shape[1] < min_size:
            continue
            
        face_images.append(face)
    
    return face_images


def extract_faces_from_video(video_path, output_dir, detector, sample_rate=30, max_frames=100):
    """
    Extract faces from video frames
    
    Args:
        video_path: Path to video file
        output_dir: Output directory
        detector: Face detector
        sample_rate: Sample one frame every N frames
        max_frames: Maximum number of frames to process
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
    face_count = 0
    
    while cap.isOpened():
        # Get next frame
        ret, frame = cap.read()
        
        if not ret:
            break
            
        # Process sampled frames
        if frame_idx in sample_indices:
            # Extract faces
            faces = extract_faces(frame, detector)
            
            # Save faces
            for face in faces:
                face_path = os.path.join(output_dir, f"frame_{frame_idx:06d}_face_{face_count:03d}.png")
                cv2.imwrite(face_path, face)
                face_count += 1
                
        frame_idx += 1
        
        # Stop if we've processed enough frames
        if frame_idx >= frame_count:
            break
            
    # Release video
    cap.release()
    
    return face_count


def setup_face_detector(device='cuda'):
    """Create MTCNN face detector"""
    return MTCNN(
        image_size=160,
        margin=0,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=True,
        device=device
    )