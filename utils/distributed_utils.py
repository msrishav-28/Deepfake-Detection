import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup_distributed(rank, world_size):
    """
    Setup distributed training
    
    Args:
        rank: Process rank
        world_size: Number of processes
    """
    # Set environment variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()

def is_main_process(rank):
    """
    Check if current process is main process
    
    Args:
        rank: Process rank
        
    Returns:
        True if main process
    """
    return rank == 0

def get_world_size():
    """
    Get world size
    
    Returns:
        World size
    """
    if not dist.is_available() or not dist.is_initialized():
        return 1
    
    return dist.get_world_size()

def get_rank():
    """
    Get process rank
    
    Returns:
        Process rank
    """
    if not dist.is_available() or not dist.is_initialized():
        return 0
    
    return dist.get_rank()

def all_gather(tensor):
    """
    All-gather tensor across processes
    
    Args:
        tensor: Input tensor
        
    Returns:
        List of tensors from all processes
    """
    world_size = get_world_size()
    
    if world_size == 1:
        return [tensor]
    
    # List of tensors
    tensors_gather = [torch.ones_like(tensor) for _ in range(world_size)]
    
    # All-gather
    dist.all_gather(tensors_gather, tensor)
    
    return tensors_gather

def reduce_dict(input_dict, average=True):
    """
    Reduce dictionary across processes
    
    Args:
        input_dict: Input dictionary
        average: Whether to average values
        
    Returns:
        Reduced dictionary
    """
    world_size = get_world_size()
    
    if world_size == 1:
        return input_dict
    
    # Create dictionary of tensors
    names = []
    values = []
    
    for k, v in sorted(input_dict.items()):
        names.append(k)
        values.append(v)
    
    # Reduce tensors
    values = torch.stack(values, dim=0)
    dist.all_reduce(values)
    
    if average:
        values /= world_size
    
    # Create output dictionary
    reduced_dict = {k: v for k, v in zip(names, values)}
    
    return reduced_dict