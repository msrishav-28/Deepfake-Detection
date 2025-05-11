```python
import os
import argparse
import cv2
import torch
from tqdm import tqdm
from pathlib import Path

from data.preprocessing.face_extraction import setup_face_detector, extract_faces_from_video

def main():
    parser = argparse.ArgumentParser(description="Extract faces from FaceForensics++ videos")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing FaceForensics++ dataset")
    parser.add_argument("--output", type=str, required=True, help="Output directory for extracted faces")
    parser.add_argument("--sample_rate", type=int, default=30, help="Sample one frame every N frames")
    parser.add_argument("--max_frames", type=int, default=100, help="Maximum number of frames to process per video")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use for face detection")
    args = parser.parse_args()
    
    # Setup face detector
    detector = setup_face_detector(device=args.device)
    
    # Directories to process
    methods = ["original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    
    for method in methods:
        if method == "original":
            input_dir = os.path.join(args.input, "original_sequences/youtube/c40/videos")
            output_dir = os.path.join(args.output, "original")
        else:
            input_dir = os.path.join(args.input, f"manipulated_sequences/{method}/c40/videos")
            output_dir = os.path.join(args.output, method)
        
        print(f"Processing {method} videos...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Get video files
        video_files = list(Path(input_dir).glob("*.mp4"))
        
        for video_file in tqdm(video_files):
            # Create output directory for this video
            video_name = video_file.stem
            video_output_dir = os.path.join(output_dir, video_name)
            os.makedirs(video_output_dir, exist_ok=True)
            
            # Extract faces
            face_count = extract_faces_from_video(
                str(video_file),
                video_output_dir,
                detector,
                sample_rate=args.sample_rate,
                max_frames=args.max_frames
            )
            
            if face_count == 0:
                print(f"WARNING: No faces detected in {video_file}")

if __name__ == "__main__":
    main()