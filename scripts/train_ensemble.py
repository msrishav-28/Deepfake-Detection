import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
import numpy as np
import random
from datetime import datetime
import json

from models.ensemble import DeepfakeEnsemble
from models.efficientnet import EfficientNetB4Detector
from models.resnet import ResNet50Detector
from models.xception import XceptionDetector
from data.datasets import DeepfakeDataset
from utils.metrics import compute_metrics
from utils.logger import setup_logger
from utils.checkpointing import save_checkpoint, load_checkpoint

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss / (pbar.n + 1), 'acc': 100 * correct / total})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = running_loss / len(dataloader)
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    
    return val_loss, metrics

def initialize_ensemble(model_paths, device):
    """Initialize ensemble model with pre-trained models"""
    models = []
    
    for model_path in model_paths:
        # Determine model type from path
        if "efficientnet" in model_path.lower():
            model = EfficientNetB4Detector(num_classes=2)
        elif "resnet" in model_path.lower():
            model = ResNet50Detector(num_classes=2)
        elif "xception" in model_path.lower():
            model = XceptionDetector(num_classes=2)
        else:
            raise ValueError(f"Cannot determine model type from path: {model_path}")
        
        # Load pre-trained weights
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        models.append(model)
    
    # Create ensemble
    ensemble = DeepfakeEnsemble(models, device=device)
    
    return ensemble

def main():
    parser = argparse.ArgumentParser(description="Train ensemble deepfake detection model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing processed face data")
    parser.add_argument("--model_paths", type=str, nargs="+", required=True, 
                        help="Paths to pre-trained model checkpoints")
    parser.add_argument("--output_dir", type=str, default="./trained_models", 
                        help="Directory to save trained ensemble model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=0.00005, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer")
    parser.add_argument("--dataset", type=str, default="combined", 
                        choices=["faceforensics", "celebdf", "combined"], 
                        help="Dataset to use for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use for training")
    parser.add_argument("--freeze_base_models", action="store_true", 
                        help="Freeze base models weights during training")
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"ensemble_{args.dataset}_{timestamp}"
    model_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(os.path.join(model_dir, "training.log"))
    logger.info(f"Training ensemble model on {args.dataset} dataset")
    logger.info(f"Using models: {args.model_paths}")
    logger.info(f"Args: {args}")
    
    # Save configuration
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # Data transforms
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets and dataloaders
    train_dataset = DeepfakeDataset(
        root_dir=args.data_dir, 
        dataset_type=args.dataset,
        split="train",
        transform=train_transform
    )
    
    val_dataset = DeepfakeDataset(
        root_dir=args.data_dir, 
        dataset_type=args.dataset,
        split="val",
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Initialize ensemble model
    model = initialize_ensemble(args.model_paths, args.device)
    
    # Freeze base models if requested
    if args.freeze_base_models:
        logger.info("Freezing base models weights")
        model.freeze_base_models()
    
    model = model.to(args.device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Training loop
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        logger.info(f"Epoch {epoch}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, args.device)
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, args.device)
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%, "
                   f"Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, "
                   f"F1: {val_metrics['f1_score']:.4f}, AUC: {val_metrics['auc']:.4f}")
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save checkpoint
        is_best = val_metrics['accuracy'] > best_val_acc
        if is_best:
            best_val_acc = val_metrics['accuracy']
        
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_metrics': val_metrics,
        }, is_best, os.path.join(model_dir, f"checkpoint_epoch_{epoch}.pth"))
    
    logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()