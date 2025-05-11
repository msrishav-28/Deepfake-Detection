# training/schedulers/scheduler_factory.py
import torch.optim as optim
import math

def create_scheduler(optimizer, config, num_epochs):
    """
    Create learning rate scheduler
    
    Args:
        optimizer: Optimizer
        config: Scheduler configuration
        num_epochs: Number of epochs
        
    Returns:
        Scheduler
    """
    scheduler_type = config['type'].lower()
    
    if scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get('step_size', 10),
            gamma=config.get('gamma', 0.1)
        )
    
    elif scheduler_type == 'multistep':
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.get('milestones', [15, 30, 45]),
            gamma=config.get('gamma', 0.1)
        )
    
    elif scheduler_type == 'cosine':
        # Cosine annealing with warmup
        warmup_epochs = config.get('warmup_epochs', 0)
        min_lr = config.get('min_lr', 0.0)
        
        if warmup_epochs > 0:
            return WarmupCosineScheduler(
                optimizer,
                warmup_epochs=warmup_epochs,
                max_epochs=num_epochs,
                min_lr=min_lr
            )
        else:
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_epochs,
                eta_min=min_lr
            )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class WarmupCosineScheduler(optim.lr_scheduler._LRScheduler):
    """Warmup Cosine Scheduler"""
    
    def __init__(
        self,
        optimizer,
        warmup_epochs,
        max_epochs,
        min_lr=0.0,
        last_epoch=-1
    ):
        """
        Args:
            optimizer: Optimizer
            warmup_epochs: Number of warmup epochs
            max_epochs: Maximum number of epochs
            min_lr: Minimum learning rate
            last_epoch: Last epoch
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Get learning rate"""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            progress = min(1.0, progress)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_factor for base_lr in self.base_lrs]