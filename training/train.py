# training/train.py
import os
import yaml
import torch
import argparse
from torch.utils.data import DataLoader

from data.datasets.faceforensics import FaceForensicsDataset
from data.datasets.celebdf import CelebDFDataset
from models.model_zoo.model_factory import create_model
from .trainer import Trainer
from utils.config_utils import load_config


def train_model(config_path, output_dir):
    """
    Train model from config
    
    Args:
        config_path: Path to config file
        output_dir: Output directory
    """
    # Load config
    config = load_config(config_path)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets
    if config['dataset'] == 'faceforensics':
        train_dataset = FaceForensicsDataset(
            root=config['data']['faceforensics_root'],
            split='train',
            img_size=config['data']['img_size'],
            transform=True,
            methods=config['data'].get('methods', None)
        )
        
        val_dataset = FaceForensicsDataset(
            root=config['data']['faceforensics_root'],
            split='val',
            img_size=config['data']['img_size'],
            transform=False,
            methods=config['data'].get('methods', None)
        )
    elif config['dataset'] == 'celebdf':
        train_dataset = CelebDFDataset(
            root=config['data']['celebdf_root'],
            split='train',
            img_size=config['data']['img_size'],
            transform=True
        )
        
        val_dataset = CelebDFDataset(
            root=config['data']['celebdf_root'],
            split='val',
            img_size=config['data']['img_size'],
            transform=False
        )
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    # Create model
    model = create_model(
        model_type=config['model']['type'],
        **config['model']['params']
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config['training'],
        device=device,
        experiment_dir=output_dir
    )
    
    # Train model
    best_metrics = trainer.train()
    
    # Save best metrics
    metrics_path = os.path.join(output_dir, 'best_metrics.yaml')
    with open(metrics_path, 'w') as f:
        yaml.dump(best_metrics, f)
    
    return best_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Train model
    train_model(args.config, args.output)