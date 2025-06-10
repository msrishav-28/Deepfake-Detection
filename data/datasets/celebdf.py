# data/datasets/celebdf.py
import os
import glob
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from ..preprocessing.normalization import get_train_transforms, get_test_transforms
from sklearn.model_selection import train_test_split

class CelebDFDataset(Dataset):
    """Celeb-DF dataset"""
    
    def __init__(self, root, split='train', img_size=224, transform=None):
        """
        Args:
            root: Dataset root directory
            split: Dataset split (train, val, test)
            img_size: Image size
            transform: Additional transforms
        """
        self.root = root
        self.split = split
        self.img_size = img_size
        
        # Set transforms
        if transform is None:
            self.transform = get_train_transforms(img_size) if split == 'train' else get_test_transforms(img_size)
        else:
            self.transform = transform
            
        # Load dataset
        self.samples = self._load_dataset()
        self._verify_dataset_structure()
    
    def _load_dataset(self):
        """Load dataset samples with stratified splitting"""
        # Real samples - FIXED PATH
        real_dir = os.path.join(self.root, "real")  # Changed from "extracted_faces/real"
        real_samples = glob.glob(os.path.join(real_dir, "**/*.png"), recursive=True)
        real_samples = [(sample, 0) for sample in real_samples]  # 0 = real
        
        # Fake samples - FIXED PATH
        fake_dir = os.path.join(self.root, "fake")  # Changed from "extracted_faces/fake"
        fake_samples = glob.glob(os.path.join(fake_dir, "**/*.png"), recursive=True)
        fake_samples = [(sample, 1) for sample in fake_samples]  # 1 = fake
            
        # Combine all samples
        all_samples = real_samples + fake_samples
        
        # Separate paths and labels for stratified splitting
        paths = [sample[0] for sample in all_samples]
        labels = [sample[1] for sample in all_samples]
        
        print(f"Total samples: {len(all_samples)}")
        print(f"Real samples: {len(real_samples)}")
        print(f"Fake samples: {len(fake_samples)}")
        
        # STRATIFIED SPLITTING - This fixes the main issue!
        if len(all_samples) == 0:
            return []
            
        # Split into train/temp, then temp into val/test
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            paths, labels, 
            test_size=0.3,  # 30% for val+test
            stratify=labels,  # This ensures balanced splits!
            random_state=42
        )
        
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels,
            test_size=0.5,  # Split the 30% equally: 15% val, 15% test
            stratify=temp_labels,
            random_state=42
        )
        
        # Return appropriate split
        if self.split == 'train':
            samples = list(zip(train_paths, train_labels))
            print(f"Train set: {len(samples)} samples")
            print(f"  - Real: {train_labels.count(0)}")
            print(f"  - Fake: {train_labels.count(1)}")
            return samples
        elif self.split == 'val':
            samples = list(zip(val_paths, val_labels))
            print(f"Validation set: {len(samples)} samples")
            print(f"  - Real: {val_labels.count(0)}")
            print(f"  - Fake: {val_labels.count(1)}")
            return samples
        else:  # test
            samples = list(zip(test_paths, test_labels))
            print(f"Test set: {len(samples)} samples")
            print(f"  - Real: {test_labels.count(0)}")
            print(f"  - Fake: {test_labels.count(1)}")
            return samples
    
    def _verify_dataset_structure(self):
        """Verify dataset directory structure"""
        required_dirs = [
            os.path.join(self.root, "real"),   # FIXED - removed "extracted_faces/"
            os.path.join(self.root, "fake")    # FIXED - removed "extracted_faces/"
        ]
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                raise ValueError(f"Required directory not found: {dir_path}")
    
    def __len__(self):
        """Dataset length"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get dataset item"""
        max_retries = 3
        for retry in range(max_retries):
            try:
                img_path, label = self.samples[idx]
                
                # Check if file exists
                if not os.path.exists(img_path):
                    # Try another random sample
                    idx = np.random.randint(0, len(self.samples))
                    continue
                    
                # Load image
                img = cv2.imread(img_path)
                if img is None:
                    idx = np.random.randint(0, len(self.samples))
                    continue
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Apply transforms
                if self.transform is not None:
                    augmented = self.transform(image=img)
                    img = augmented['image']
                
                return img, float(label)
            except Exception as e:
                if retry == max_retries - 1:
                    # Return a default black image as last resort
                    print(f"Error loading sample {idx}: {str(e)}")
                    img = torch.zeros(3, self.img_size, self.img_size)
                    return img, float(0)
                idx = np.random.randint(0, len(self.samples))
        
    def _balance_samples(self, real_samples, fake_samples):
        """Balance dataset samples"""
        min_samples = min(len(real_samples), len(fake_samples))
        if len(real_samples) > min_samples:
            real_samples = random.sample(real_samples, min_samples)
        if len(fake_samples) > min_samples:
            fake_samples = random.sample(fake_samples, min_samples)
        return real_samples, fake_samples