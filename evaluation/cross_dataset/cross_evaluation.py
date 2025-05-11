# evaluation/cross_dataset/cross_evaluation.py
import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets.faceforensics import FaceForensicsDataset
from data.datasets.celebdf import CelebDFDataset
from models.model_zoo.model_factory import create_model
from ..metrics.classification_metrics import calculate_metrics
from utils.logging_utils import AverageMeter

def cross_dataset_evaluation(model, source_dataset, target_dataset, device, batch_size=32):
    """
    Evaluate model on a target dataset
    
    Args:
        model: Model to evaluate
        source_dataset: Source dataset name
        target_dataset: Target dataset
        device: Device to use
        batch_size: Batch size
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    # Create dataloader
    dataloader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize metrics
    loss_meter = AverageMeter()
    all_preds = []
    all_targets = []
    
    # Evaluation loop
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc=f"Evaluating {source_dataset} -> {target_dataset.__class__.__name__}"):
            # Move to device
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Update metrics
            all_preds.extend(outputs.detach().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    metrics = calculate_metrics(all_preds, all_targets)
    
    return metrics

def evaluate_cross_datasets(config, output_dir):
    """
    Evaluate model on cross datasets
    
    Args:
        config: Configuration dictionary
        output_dir: Output directory
        
    Returns:
        Dictionary of metrics
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    if config['model']['type'] == 'fusion':
        # Load fusion model
        raise NotImplementedError("Fusion model evaluation not implemented yet")
    else:
        # Create model
        model = create_model(
            model_type=config['model']['type'],
            **config['model']['params']
        )
        
        # Load checkpoint
        checkpoint = torch.load(config['model']['checkpoint'], map_location=device)
        model.load_state_dict(checkpoint['model'])
    
    # Move model to device
    model = model.to(device)
    
    # Create datasets
    ff_dataset = FaceForensicsDataset(
        root=config['data']['faceforensics_root'],
        split='test',
        img_size=config['data']['img_size'],
        transform=False,
        methods=config['data'].get('methods', None)
    )
    
    celebdf_dataset = CelebDFDataset(
        root=config['data']['celebdf_root'],
        split='test',
        img_size=config['data']['img_size'],
        transform=False
    )
    
    # Evaluate on source dataset
    if config['cross_dataset']['source_dataset'] == 'faceforensics':
        source_metrics = cross_dataset_evaluation(
            model=model,
            source_dataset='faceforensics',
            target_dataset=ff_dataset,
            device=device,
            batch_size=config['evaluation']['batch_size']
        )
    else:
        source_metrics = cross_dataset_evaluation(
            model=model,
            source_dataset='celebdf',
            target_dataset=celebdf_dataset,
            device=device,
            batch_size=config['evaluation']['batch_size']
        )
    
    # Evaluate on target dataset
    if config['cross_dataset']['target_dataset'] == 'faceforensics':
        target_metrics = cross_dataset_evaluation(
            model=model,
            source_dataset=config['cross_dataset']['source_dataset'],
            target_dataset=ff_dataset,
            device=device,
            batch_size=config['evaluation']['batch_size']
        )
    else:
        target_metrics = cross_dataset_evaluation(
            model=model,
            source_dataset=config['cross_dataset']['source_dataset'],
            target_dataset=celebdf_dataset,
            device=device,
            batch_size=config['evaluation']['batch_size']
        )
    
    # Calculate generalization gap
    generalization_gap = {
        'accuracy': source_metrics['accuracy'] - target_metrics['accuracy'],
        'auc': source_metrics['auc'] - target_metrics['auc'],
        'eer': target_metrics['eer'] - source_metrics['eer']
    }
    
    # Save results
    results = {
        'source_dataset': config['cross_dataset']['source_dataset'],
        'target_dataset': config['cross_dataset']['target_dataset'],
        'source_metrics': source_metrics,
        'target_metrics': target_metrics,
        'generalization_gap': generalization_gap
    }
    
    # Save to file
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'cross_evaluation_results.yaml'), 'w') as f:
        yaml.dump(results, f)
    
    return results