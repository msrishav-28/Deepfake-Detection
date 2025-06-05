# data/datasets/celebdf.py
import os
import glob
import random
import numpy as np
import cv2
from torch.utils.data import Dataset
from ..preprocessing.normalization import get_train_transforms, get_test_transforms

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
        """Load dataset samples"""
        samples = []
        
        # Real samples
        real_dir = os.path.join(self.root, "extracted_faces/real")
        real_samples = glob.glob(os.path.join(real_dir, "**/*.png"), recursive=True)
        real_samples = [(sample, 0) for sample in real_samples]  # 0 = real
        
        # Fake samples
        fake_dir = os.path.join(self.root, "extracted_faces/fake")
        fake_samples = glob.glob(os.path.join(fake_dir, "**/*.png"), recursive=True)
        fake_samples = [(sample, 1) for sample in fake_samples]  # 1 = fake
            
        # Combine samples
        all_samples = real_samples + fake_samples
        
        # Split dataset
        random.seed(42)
        random.shuffle(all_samples)
        
        # Calculate split sizes
        total_size = len(all_samples)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        
        # Split samples
        if self.split == 'train':
            return all_samples[:train_size]
        elif self.split == 'val':
            return all_samples[train_size:train_size + val_size]
        else:  # test
            return all_samples[train_size + val_size:]
    
    def _verify_dataset_structure(self):
        """Verify dataset directory structure"""
        required_dirs = [
            os.path.join(self.root, "extracted_faces/real"),
            os.path.join(self.root, "extracted_faces/fake")
        ]
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                raise ValueError(f"Required directory not found: {dir_path}")
    
    def __len__(self):
        """Dataset length"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get dataset item"""
        try:
            img_path, label = self.samples[idx]
            
            # Check if file exists
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
                
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transform(img)
            
            return img, float(label)
        except Exception as e:
            print(f"Error loading sample {idx}: {str(e)}")
            # Return a default or skip sample
            return None
        
    def _balance_samples(self, real_samples, fake_samples):
        """Balance dataset samples"""
        min_samples = min(len(real_samples), len(fake_samples))
        if len(real_samples) > min_samples:
            real_samples = random.sample(real_samples, min_samples)
        if len(fake_samples) > min_samples:
            fake_samples = random.sample(fake_samples, min_samples)
        return real_samples, fake_samples