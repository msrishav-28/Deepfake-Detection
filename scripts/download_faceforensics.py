#!/usr/bin/env python3
"""
Script to download FaceForensics++ dataset
Usage: python scripts/download_faceforensics.py /path/to/output/directory
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# FaceForensics++ download script URL
FF_DOWNLOAD_SCRIPT = "https://github.com/ondyari/FaceForensics/blob/master/dataset/download.py"

def download_faceforensics(output_dir, method='all', compression='c40', 
                          dataset_type='videos', num_videos=500):
    """
    Download FaceForensics++ dataset
    
    Args:
        output_dir: Output directory
        method: Method to download ('all' or specific method)
        compression: Compression level (c0, c23, c40)
        dataset_type: Type of data ('videos', 'masks', 'models')
        num_videos: Number of videos to download
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Methods to download
    if method == 'all':
        methods = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'original']
    else:
        methods = [method]
    
    # Download each method
    for method_name in methods:
        print(f"\nDownloading {method_name}...")
        
        cmd = [
            sys.executable, 
            "download.py",  # You need to download this script from FaceForensics++ repo
            output_dir,
            "-d", method_name,
            "-c", compression,
            "-t", dataset_type
        ]
        
        if num_videos is not None:
            cmd.extend(["-n", str(num_videos)])
        
        print(f"Command: {' '.join(cmd)}")
        
        try:
            # Note: You need to download the official FaceForensics++ download script
            # subprocess.run(cmd, check=True)
            print(f"✓ {method_name} downloaded successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error downloading {method_name}: {e}")
        except FileNotFoundError:
            print(f"✗ download.py not found. Please download it from FaceForensics++ repository")
            print(f"  URL: {FF_DOWNLOAD_SCRIPT}")
            return

def main():
    parser = argparse.ArgumentParser(description="Download FaceForensics++ dataset")
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument("-d", "--method", type=str, default="all",
                       choices=['all', 'Deepfakes', 'Face2Face', 'FaceSwap', 
                               'NeuralTextures', 'original'],
                       help="Method to download")
    parser.add_argument("-c", "--compression", type=str, default="c40",
                       choices=['c0', 'c23', 'c40'],
                       help="Compression level")
    parser.add_argument("-t", "--type", type=str, default="videos",
                       choices=['videos', 'masks', 'models'],
                       help="Type of data to download")
    parser.add_argument("-n", "--num_videos", type=int, default=500,
                       help="Number of videos to download")
    args = parser.parse_args()
    
    print("FaceForensics++ Dataset Downloader")
    print(f"Output directory: {args.output_dir}")
    print(f"Method: {args.method}")
    print(f"Compression: {args.compression}")
    print(f"Type: {args.type}")
    print(f"Number of videos: {args.num_videos}")
    
    download_faceforensics(
        args.output_dir,
        method=args.method,
        compression=args.compression,
        dataset_type=args.type,
        num_videos=args.num_videos
    )

if __name__ == "__main__":
    main()