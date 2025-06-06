import os
import argparse
import cv2
import torch
from tqdm import tqdm
from pathlib import Path

from data.preprocessing.face_extraction import setup_face_detector, extract_faces_from_video

def main():
    parser = argparse.ArgumentParser(description="Extract faces from Celeb-DF videos")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing Celeb-DF dataset")
    parser.add_argument("--output", type=str, required=True, help="Output directory for extracted faces")
    parser.add_argument("--sample_rate", type=int, default=30, help="Sample one frame every N frames")
    parser.add_argument("--max_frames", type=int, default=100, help="Maximum number of frames to process per video")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use for face detection")
    args = parser.parse_args()
    
    # Setup face detector
    detector = setup_face_detector(device=args.device)

    # Process real videos from both Celeb-real and YouTube-real
    real_dirs = ["Celeb-real", "YouTube-real"]  # Add YouTube-real
    for real_dir in real_dirs:
        real_dir_path = os.path.join(args.input, real_dir)
        if os.path.exists(real_dir_path):
            real_output_dir = os.path.join(args.output, "real")
            os.makedirs(real_output_dir, exist_ok=True)
            
            print(f"Processing {real_dir} videos...")
            process_videos(real_dir_path, real_output_dir, detector, args)

    # Process fake videos
    fake_dir = os.path.join(args.input, "Celeb-synthesis")
    fake_output_dir = os.path.join(args.output, "fake")
    os.makedirs(fake_output_dir, exist_ok=True)

    print("Processing fake videos...")
    process_videos(fake_dir, fake_output_dir, detector, args)

def process_videos(input_dir, output_dir, detector, args):
    # Get video files
    video_files = []
    for ext in ["*.mp4", "*.avi", "*.mov"]:
        video_files.extend(list(Path(input_dir).glob(ext)))
    
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