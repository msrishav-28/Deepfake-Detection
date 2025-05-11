# data/datasets/custom_dataset.py
import torch
from torch.utils.data import Dataset

class DeepfakeDataset(Dataset):
    """Custom deepfake dataset combining multiple sources"""
    
    def __init__(self, datasets):
        """
        Args:
            datasets: List of datasets to combine
        """
        self.datasets = datasets
        self.dataset_lengths = [len(dataset) for dataset in datasets]
        self.cumulative_lengths = [0]
        
        # Calculate cumulative lengths
        for length in self.dataset_lengths:
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + length)
    
    def __len__(self):
        """Dataset length"""
        return self.cumulative_lengths[-1]
    
    def __getitem__(self, idx):
        """Get dataset item"""
        # Find dataset index
        dataset_idx = 0
        while idx >= self.cumulative_lengths[dataset_idx + 1]:
            dataset_idx += 1
            
        # Get item from dataset
        item_idx = idx - self.cumulative_lengths[dataset_idx]
        return self.datasets[dataset_idx][item_idx]