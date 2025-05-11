# data/datasets/faceforensics.py
import os
import glob
import random
import numpy as np
import cv2
from torch.utils.data import Dataset
from ..preprocessing.normalization import get_train_transforms, get_test_transforms

class FaceForensicsDataset(Dataset):
    """FaceForensics++ dataset"""
    
    def __init__(self, root, split='train', img_size=224, transform=None, methods=None):
        """
        Args:
            root: Dataset root directory
            split: Dataset split (train, val, test)
            img_size: Image size
            transform: Additional transforms
            methods: List of manipulation methods to include
        """
        self.root = root
        self.split = split
        self.img_size = img_size
        
        # Set transforms
        if transform is None:
            self.transform = get_train_transforms(img_size) if split == 'train' else get_test_transforms(img_size)
        else:
            self.transform = transform
            
        # Set manipulation methods
        if methods is None:
            self.methods = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
        else:
            self.methods = methods
            
        # Load dataset
        self.samples = self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset samples"""
        samples = []
        
        # Real samples
        real_dir = os.path.join(self.root, "extracted_faces/original")
        real_samples = glob.glob(os.path.join(real_dir, "**/*.png"), recursive=True)
        real_samples = [(sample, 0) for sample in real_samples]  # 0 = real
        
        # Fake samples
        fake_samples = []
        for method in self.methods:
            fake_dir = os.path.join(self.root, f"extracted_faces/{method}")
            method_samples = glob.glob(os.path.join(fake_dir, "**/*.png"), recursive=True)
            method_samples = [(sample, 1) for sample in method_samples]  # 1 = fake
            fake_samples.extend(method_samples)
            
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
    
    def __len__(self):
        """Dataset length"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get dataset item"""
        # Get sample
        img_path, label = self.samples[idx]
        
        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply transform
        img = self.transform(img)
        
        return img, float(label)