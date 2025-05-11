# training/trainer.py
import os
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from .optimizers.optimizer_factory import create_optimizer
from .schedulers.scheduler_factory import create_scheduler
from .callbacks.early_stopping import EarlyStopping
from .losses.focal_loss import BinaryFocalLoss
from utils.logging_utils import AverageMeter
from evaluation.metrics.classification_metrics import calculate_metrics

class Trainer:
    """Model trainer"""
    
    def __init__(self, model, train_loader, val_loader, config, device, experiment_dir):
        """
        Args:
            model: Model to train
            train_loader: Training dataloader
            val_loader: Validation dataloader
            config: Training configuration
            device: Device to use
            experiment_dir: Experiment directory
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.experiment_dir = experiment_dir
        
        # Create optimizer
        self.optimizer = create_optimizer(
            self.model.parameters(),
            config['optimizer']
        )
        
        # Create scheduler
        self.scheduler = create_scheduler(
            self.optimizer,
            config['scheduler'],
            config['epochs']
        )
        
        # Create criterion
        self.criterion = BinaryFocalLoss(alpha=0.25, gamma=2.0)
        
        # Create early stopping
        self.early_stopping = EarlyStopping(
            patience=config['early_stopping']['patience'],
            min_delta=config['early_stopping']['min_delta'],
            mode='min'
        )
        
        # Create summary writer
        self.writer = SummaryWriter(os.path.join(experiment_dir, 'logs'))
        
        # Create checkpoint directory
        self.checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize counters
        self.epoch = 0
        self.global_step = 0
        
        # Initialize best metrics
        self.best_metrics = {
            'loss': float('inf'),
            'accuracy': 0.0,
            'auc': 0.0,
            'eer': float('inf')
        }
        
        # Initialize scaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config.get('use_amp', False) else None
    
    def train(self):
        """Train model"""
        print(f"Starting training for {self.config['epochs']} epochs")
        
        # Start training
        for epoch in range(self.epoch, self.config['epochs']):
            self.epoch = epoch
            
            # Train for one epoch
            train_metrics = self._train_epoch()
            
            # Validate
            val_metrics = self._validate()
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Log metrics
            self._log_metrics(train_metrics, val_metrics)
            
            # Save checkpoint
            self._save_checkpoint()
            
            # Check early stopping
            if self.early_stopping(val_metrics['loss']):
                print(f"Early stopping at epoch {epoch}")
                break
        
        print("Training completed")
        return self.best_metrics
    
    def _train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        # Initialize metrics
        loss_meter = AverageMeter()
        all_preds = []
        all_targets = []
        
        # Start epoch
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch} [Train]")
        
        # Training loop
        for batch_idx, (images, targets) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Mixed precision training
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    # Forward pass
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets.unsqueeze(1))
                    
                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.get('grad_clip', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                
                # Update weights
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, targets.unsqueeze(1))
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.config.get('grad_clip', 0) > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                
                # Update weights
                self.optimizer.step()
            
            # Update metrics
            loss_meter.update(loss.item(), images.size(0))
            all_preds.extend(outputs.detach().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss_meter.avg})
            
            # Log metrics
            if batch_idx % self.config.get('log_interval', 50) == 0:
                self.writer.add_scalar('train/loss', loss_meter.avg, self.global_step)
                self.global_step += 1
        
        # Calculate epoch metrics
        all_preds = np.array(all_preds).flatten()
        all_targets = np.array(all_targets).flatten()
        metrics = calculate_metrics(all_preds, all_targets)
        metrics['loss'] = loss_meter.avg
        
        return metrics
    
    def _validate(self):
        """Validate model"""
        self.model.eval()
        
        # Initialize metrics
        loss_meter = AverageMeter()
        all_preds = []
        all_targets = []
        
        # Start validation
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.epoch} [Val]")
        
        # Validation loop
        with torch.no_grad():
            for images, targets in pbar:
                # Move to device
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, targets.unsqueeze(1))
                
                # Update metrics
                loss_meter.update(loss.item(), images.size(0))
                all_preds.extend(outputs.detach().cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({'loss': loss_meter.avg})
        
        # Calculate epoch metrics
        all_preds = np.array(all_preds).flatten()
        all_targets = np.array(all_targets).flatten()
        metrics = calculate_metrics(all_preds, all_targets)
        metrics['loss'] = loss_meter.avg
        
        # Update best metrics
        if metrics['auc'] > self.best_metrics['auc']:
            self.best_metrics = metrics.copy()
            self._save_checkpoint(is_best=True)
        
        return metrics
    
    def _log_metrics(self, train_metrics, val_metrics):
        """Log metrics"""
        # Print metrics
        print(f"Epoch {self.epoch}")
        print(f"Train: loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.4f}, auc={train_metrics['auc']:.4f}, eer={train_metrics['eer']:.4f}")
        print(f"Val  : loss={val_metrics['loss']:.4f}, acc={val_metrics['accuracy']:.4f}, auc={val_metrics['auc']:.4f}, eer={val_metrics['eer']:.4f}")
        
        # Log to TensorBoard
        for k, v in train_metrics.items():
            self.writer.add_scalar(f'train/{k}', v, self.epoch)
        
        for k, v in val_metrics.items():
            self.writer.add_scalar(f'val/{k}', v, self.epoch)
    
    def _save_checkpoint(self, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_metrics': self.best_metrics
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            
        # Save epoch checkpoint
        epoch_path = os.path.join(self.checkpoint_dir, f'epoch_{self.epoch}.pth')
        torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model
        self.model.load_state_dict(checkpoint['model'])
        
        # Load optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Load scheduler
        if self.scheduler and checkpoint['scheduler']:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        
        # Load counters
        self.epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        
        # Load best metrics
        self.best_metrics = checkpoint['best_metrics']
        
        return self.epoch