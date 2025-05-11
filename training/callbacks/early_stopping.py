# training/callbacks/early_stopping.py
import numpy as np

class EarlyStopping:
    """Early stopping callback"""
    
    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        """
        Args:
            patience: Number of epochs to wait
            min_delta: Minimum change in monitored value
            mode: 'min' or 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.is_better = None
        self._init_is_better(mode)
    
    def _init_is_better(self, mode):
        """Initialize is_better function"""
        if mode == 'min':
            self.is_better = lambda score, best_score: score < best_score - self.min_delta
        elif mode == 'max':
            self.is_better = lambda score, best_score: score > best_score + self.min_delta
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def __call__(self, score):
        """
        Args:
            score: Monitored score
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience