# training/optimizers/optimizer_factory.py
import torch.optim as optim

def create_optimizer(parameters, config):
    """
    Create optimizer
    
    Args:
        parameters: Model parameters
        config: Optimizer configuration
        
    Returns:
        Optimizer
    """
    optimizer_type = config['type'].lower()
    
    if optimizer_type == 'sgd':
        return optim.SGD(
            parameters,
            lr=config['lr'],
            momentum=config.get('momentum', 0.9),
            weight_decay=config.get('weight_decay', 0.0),
            nesterov=config.get('nesterov', False)
        )
    
    elif optimizer_type == 'adam':
        return optim.Adam(
            parameters,
            lr=config['lr'],
            betas=config.get('betas', (0.9, 0.999)),
            weight_decay=config.get('weight_decay', 0.0)
        )
    
    elif optimizer_type == 'adamw':
        return optim.AdamW(
            parameters,
            lr=config['lr'],
            betas=config.get('betas', (0.9, 0.999)),
            weight_decay=config.get('weight_decay', 0.01)
        )
    
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")