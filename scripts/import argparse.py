import argparse
import torch
from tqdm import tqdm
import wandb
from dataset import CelebDFDataset, FaceForensicsDataset  # Assuming these are your dataset classes
from transforms import train_transform, val_transform  # Assuming these are your transform functions
from early_stopping import EarlyStopping  # Assuming you have an EarlyStopping class
import logging

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    try:
        for inputs, labels in pbar:
            # Clear GPU cache if needed
            if device == 'cuda':
                torch.cuda.empty_cache()
                
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Add gradient scaling for mixed precision training
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            loss.backward()
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1), 'acc': 100 * correct / total})
            
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        raise e
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def main():
    parser = argparse.ArgumentParser(description="Train individual deepfake detection models")
    # ...existing argument parsing code...
    
    try:
        # Initialize wandb
        wandb.init(
            project="deepfake-detection",
            name=model_name,
            config=vars(args)
        )
        
        # Verify CUDA availability
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please use CPU instead.")
        
        # Create dataset based on type
        if args.dataset == "celebdf":
            dataset_class = CelebDFDataset
        elif args.dataset == "faceforensics":
            dataset_class = FaceForensicsDataset
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")
        
        # Create datasets with proper error handling
        try:
            train_dataset = dataset_class(
                root=args.data_dir,
                split="train",
                transform=train_transform
            )
            val_dataset = dataset_class(
                root=args.data_dir,
                split="val",
                transform=val_transform
            )
        except Exception as e:
            logger.error(f"Failed to create datasets: {str(e)}")
            raise e
        
        # Add early stopping
        early_stopping = EarlyStopping(patience=5, min_delta=0.001)
        
        # Training loop
        for epoch in range(1, args.epochs + 1):
            try:
                train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, args.device)
                val_loss, val_metrics = validate(model, val_loader, criterion, args.device)
                
                # Log metrics to wandb
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    **val_metrics
                })
                
                # Early stopping check
                if early_stopping(val_loss):
                    logger.info("Early stopping triggered")
                    break
                
            except Exception as e:
                logger.error(f"Error during epoch {epoch}: {str(e)}")
                raise e
                
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise e
    
    finally:
        # Cleanup
        wandb.finish()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Entry point
if __name__ == "__main__":
    main()