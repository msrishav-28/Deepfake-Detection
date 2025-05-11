# training/callbacks/model_checkpoint.py
import os
import torch

class ModelCheckpoint:
    """Model checkpoint callback"""
    
    def __init__(self, filepath, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=False):
        """
        Args:
            filepath: Filepath to save checkpoint
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_best_only: Save only the best model
            save_weights_only: Save only weights
        """
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        
        self.best_score = None
        self.is_better = None
        self._init_is_better(mode)
    
    def _init_is_better(self, mode):
        """Initialize is_better function"""
        if mode == 'min':
            self.is_better = lambda score, best_score: score < best_score
        elif mode == 'max':
            self.is_better = lambda score, best_score: score > best_score
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def __call__(self, model, optimizer=None, epoch=0, metrics=None):
        """
        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            metrics: Current metrics
        """
        # Get monitored score
        if metrics is None or self.monitor not in metrics:
            score = float('inf') if self.mode == 'min' else float('-inf')
        else:
            score = metrics[self.monitor]
        
        # Check if score is better
        if self.best_score is None or self.is_better(score, self.best_score):
            self.best_score = score
            
            # Create checkpoint
            if self.save_weights_only:
                checkpoint = {
                    'model': model.state_dict(),
                    'epoch': epoch
                }
                
                if optimizer is not None:
                    checkpoint['optimizer'] = optimizer.state_dict()
            else:
                checkpoint = {
                    'model': model,
                    'epoch': epoch
                }
                
                if optimizer is not None:
                    checkpoint['optimizer'] = optimizer
            
            # Save checkpoint
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            torch.save(checkpoint, self.filepath)
            
            return True
        
        return False