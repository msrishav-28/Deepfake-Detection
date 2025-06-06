#!/usr/bin/env python3
"""
Script to extract and organize CelebDF dataset from zip files
"""

import os
import zipfile
import shutil
from pathlib import Path
import argparse

def extract_celebdf(input_dir, output_dir, version='v2'):
    """
    Extract CelebDF dataset from zip files
    
    Args:
        input_dir: Directory containing zip files
        output_dir: Output directory for extracted dataset
        version: CelebDF version ('v1' or 'v2')
    """
    print(f"Extracting CelebDF {version} dataset...")
    
    # Define zip file name based on version
    if version == 'v2':
        zip_file = os.path.join(input_dir, "Celeb-DF-v2.zip")
    else:
        zip_file = os.path.join(input_dir, "Celeb-DF-v1.zip")
    
    # Check if zip file exists
    if not os.path.exists(zip_file):
        raise FileNotFoundError(f"Zip file not found: {zip_file}")
    
    # Create temporary extraction directory
    temp_dir = os.path.join(output_dir, "temp_extract")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Extract zip file
    print(f"Extracting {zip_file}...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # Find the actual dataset directory
    extracted_dirs = os.listdir(temp_dir)
    if len(extracted_dirs) == 1:
        dataset_dir = os.path.join(temp_dir, extracted_dirs[0])
    else:
        # Look for Celeb-DF directory
        for dir_name in extracted_dirs:
            if 'Celeb-DF' in dir_name or 'celebdf' in dir_name.lower():
                dataset_dir = os.path.join(temp_dir, dir_name)
                break
        else:
            dataset_dir = temp_dir
    
    # Create output directory structure
    output_dataset_dir = os.path.join(output_dir, f"celebdf_{version}")
    os.makedirs(output_dataset_dir, exist_ok=True)
    
    # Expected directories in CelebDF
    expected_dirs = {
        'v2': ['Celeb-real', 'Celeb-synthesis', 'YouTube-real'],
        'v1': ['Celeb-real', 'Celeb-synthesis']
    }
    
    # Move directories to proper location
    for dir_name in expected_dirs.get(version, expected_dirs['v2']):
        src_dir = os.path.join(dataset_dir, dir_name)
        dst_dir = os.path.join(output_dataset_dir, dir_name)
        
        if os.path.exists(src_dir):
            print(f"Moving {dir_name}...")
            if os.path.exists(dst_dir):
                shutil.rmtree(dst_dir)
            shutil.move(src_dir, dst_dir)
        else:
            print(f"Warning: {dir_name} not found in extracted files")
    
    # Copy List_of_testing_videos.txt if exists
    test_list = os.path.join(dataset_dir, "List_of_testing_videos.txt")
    if os.path.exists(test_list):
        shutil.copy(test_list, output_dataset_dir)
    
    # Clean up temporary directory
    shutil.rmtree(temp_dir)
    
    print(f"CelebDF {version} dataset extracted to: {output_dataset_dir}")
    
    # Print dataset statistics
    print("\nDataset structure:")
    for root, dirs, files in os.walk(output_dataset_dir):
        level = root.replace(output_dataset_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files) - 5} more files")

def main():
    parser = argparse.ArgumentParser(description="Extract CelebDF dataset")
    parser.add_argument("--input", type=str, default="data/raw_data/celebdf",
                       help="Input directory containing zip files")
    parser.add_argument("--output", type=str, default="data/processed_data",
                       help="Output directory for extracted dataset")
    parser.add_argument("--version", type=str, default="v2", choices=['v1', 'v2'],
                       help="CelebDF version to extract")
    args = parser.parse_args()
    
    try:
        extract_celebdf(args.input, args.output, args.version)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())