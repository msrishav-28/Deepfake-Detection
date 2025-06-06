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

# Fixed imports to match project structure
from models.model_zoo.model_factory import create_model
from data.datasets.celebdf import CelebDFDataset
from data.datasets.faceforensics import FaceForensicsDataset
from evaluation.metrics.classification_metrics import calculate_metrics
from utils.logging_utils import setup_logger, AverageMeter
from utils.file_utils import ensure_dir

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(state, is_best, checkpoint_path):
    """Save checkpoint"""
    ensure_dir(os.path.dirname(checkpoint_path))
    torch.save(state, checkpoint_path)
    
    if is_best:
        best_path = os.path.join(os.path.dirname(checkpoint_path), 'model_best.pth')
        torch.save(state, best_path)

def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                if outputs.dim() == 1:
                    outputs = outputs.unsqueeze(1)
                loss = criterion(outputs, labels.float().unsqueeze(1))
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)
            loss = criterion(outputs, labels.float().unsqueeze(1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        running_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        total += labels.size(0)
        correct += (preds.squeeze() == labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss / (pbar.n + 1), 'acc': 100 * correct / total})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)
            loss = criterion(outputs, labels.float().unsqueeze(1))
            
            running_loss += loss.item()
            probs = torch.sigmoid(outputs).squeeze()
            preds = (probs > 0.5).float()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = running_loss / len(dataloader)
    metrics = calculate_metrics(np.array(all_probs), np.array(all_labels))
    
    return val_loss, metrics

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False

def main():
    parser = argparse.ArgumentParser(description="Train individual deepfake detection models")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing processed face data")
    parser.add_argument("--model_type", type=str, default="vit", 
                        choices=["vit", "deit", "swin"], 
                        help="Type of model to train")
    parser.add_argument("--output_dir", type=str, default="./trained_models", 
                        help="Directory to save trained models")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer")
    parser.add_argument("--dataset", type=str, default="celebdf", 
                        choices=["faceforensics", "celebdf"], 
                        help="Dataset to use for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use for training")
    parser.add_argument("--use_amp", action="store_true", help="Use automatic mixed precision")
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{args.model_type}_{args.dataset}_{timestamp}"
    model_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Setup logger
    logger = setup_logger("training", os.path.join(model_dir, "training.log"))
    logger.info(f"Training {args.model_type} model on {args.dataset} dataset")
    logger.info(f"Args: {args}")
    
    # Save configuration
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # Create datasets and dataloaders
    if args.dataset == "faceforensics":
        train_dataset = FaceForensicsDataset(
            root=args.data_dir, 
            split="train",
            img_size=224,
            transform=None  # Transforms are handled in the dataset
        )
        
        val_dataset = FaceForensicsDataset(
            root=args.data_dir, 
            split="val",
            img_size=224,
            transform=None
        )
    else:  # celebdf
        train_dataset = CelebDFDataset(
            root=args.data_dir, 
            split="train",
            img_size=224,
            transform=None
        )
        
        val_dataset = CelebDFDataset(
            root=args.data_dir, 
            split="val",
            img_size=224,
            transform=None
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
    
    # Initialize model
    model_params = {
        'img_size': 224,
        'patch_size': 16 if args.model_type != 'swin' else 4,
        'in_channels': 3,
        'num_classes': 1,
        'embed_dim': 768 if args.model_type != 'swin' else 96,
        'depth': 12 if args.model_type != 'swin' else [2, 2, 6, 2],
        'num_heads': 12 if args.model_type != 'swin' else [3, 6, 12, 24],
        'mlp_ratio': 4.0,
        'dropout': 0.1,
        'attn_dropout': 0.0
    }

    if args.model_type == 'swin':
        model_params['depths'] = [2, 2, 6, 2]
        model_params['num_heads'] = [3, 6, 12, 24]
        model_params['window_size'] = 7

    model = create_model(args.model_type, **model_params)
    model = model.to(args.device)
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Initialize AMP scaler if requested
    scaler = torch.cuda.amp.GradScaler() if args.use_amp and args.device == 'cuda' else None
    
    # Early stopping
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    
    # Training loop
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        logger.info(f"Epoch {epoch}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, args.device, scaler)
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, args.device)
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%, "
                   f"Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, "
                   f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Early stopping
        if early_stopping(val_loss):
            logger.info("Early stopping triggered")
            break
        
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