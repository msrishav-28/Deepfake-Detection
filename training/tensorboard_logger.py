# training/callbacks/tensorboard_logger.py
from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger:
    """TensorBoard logger callback"""
    
    def __init__(self, log_dir, log_interval=50):
        """
        Args:
            log_dir: Log directory
            log_interval: Logging interval
        """
        self.writer = SummaryWriter(log_dir)
        self.log_interval = log_interval
        self.global_step = 0
    
    def on_epoch_end(self, epoch, metrics):
        """
        Args:
            epoch: Current epoch
            metrics: Current metrics
        """
        # Log metrics
        for k, v in metrics.items():
            self.writer.add_scalar(f'epoch/{k}', v, epoch)
            
    def on_batch_end(self, batch_idx, loss, metrics=None):
        """
        Args:
            batch_idx: Current batch index
            loss: Current loss
            metrics: Current metrics
        """
        # Log loss
        if batch_idx % self.log_interval == 0:
            self.writer.add_scalar('batch/loss', loss, self.global_step)
            
            # Log metrics
            if metrics is not None:
                for k, v in metrics.items():
                    self.writer.add_scalar(f'batch/{k}', v, self.global_step)
                    
            self.global_step += 1
    
    def close(self):
        """Close writer"""
        self.writer.close()