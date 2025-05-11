# data/dataloaders/dataloader_factory.py
from torch.utils.data import DataLoader
from ..datasets import FaceForensicsDataset, CelebDFDataset, DeepfakeDataset

def create_dataloaders(config):
    """
    Create train, validation, and test dataloaders
    
    Args:
        config: Configuration dict
        
    Returns:
        Train, validation, and test dataloaders
    """
    dataloaders = {}
    
    # Create datasets
    if config["dataset"] == "faceforensics":
        train_dataset = FaceForensicsDataset(
            root=config["data"]["faceforensics_root"],
            split="train",
            img_size=config["data"]["img_size"],
            transform=None,
            methods=config["data"].get("methods", None)
        )
        
        val_dataset = FaceForensicsDataset(
            root=config["data"]["faceforensics_root"],
            split="val",
            img_size=config["data"]["img_size"],
            transform=None,
            methods=config["data"].get("methods", None)
        )
        
        test_dataset = FaceForensicsDataset(
            root=config["data"]["faceforensics_root"],
            split="test",
            img_size=config["data"]["img_size"],
            transform=None,
            methods=config["data"].get("methods", None)
        )
    
    elif config["dataset"] == "celebdf":
        train_dataset = CelebDFDataset(
            root=config["data"]["celebdf_root"],
            split="train",
            img_size=config["data"]["img_size"],
            transform=None
        )
        
        val_dataset = CelebDFDataset(
            root=config["data"]["celebdf_root"],
            split="val",
            img_size=config["data"]["img_size"],
            transform=None
        )
        
        test_dataset = CelebDFDataset(
            root=config["data"]["celebdf_root"],
            split="test",
            img_size=config["data"]["img_size"],
            transform=None
        )
    
    elif config["dataset"] == "combined":
        # Create combined datasets
        train_ff = FaceForensicsDataset(
            root=config["data"]["faceforensics_root"],
            split="train",
            img_size=config["data"]["img_size"],
            transform=None,
            methods=config["data"].get("methods", None)
        )
        
        train_celebdf = CelebDFDataset(
            root=config["data"]["celebdf_root"],
            split="train",
            img_size=config["data"]["img_size"],
            transform=None
        )
        
        val_ff = FaceForensicsDataset(
            root=config["data"]["faceforensics_root"],
            split="val",
            img_size=config["data"]["img_size"],
            transform=None,
            methods=config["data"].get("methods", None)
        )
        
        val_celebdf = CelebDFDataset(
            root=config["data"]["celebdf_root"],
            split="val",
            img_size=config["data"]["img_size"],
            transform=None
        )
        
        test_ff = FaceForensicsDataset(
            root=config["data"]["faceforensics_root"],
            split="test",
            img_size=config["data"]["img_size"],
            transform=None,
            methods=config["data"].get("methods", None)
        )
        
        test_celebdf = CelebDFDataset(
            root=config["data"]["celebdf_root"],
            split="test",
            img_size=config["data"]["img_size"],
            transform=None
        )
        
        # Combine datasets
        train_dataset = DeepfakeDataset([train_ff, train_celebdf])
        val_dataset = DeepfakeDataset([val_ff, val_celebdf])
        test_dataset = DeepfakeDataset([test_ff, test_celebdf])
    
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']}")
    
    # Create dataloaders
    dataloaders["train"] = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=True
    )
    
    dataloaders["val"] = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=True
    )
    
    dataloaders["test"] = DataLoader(
        test_dataset,
        batch_size=config["evaluation"]["batch_size"],
        shuffle=False,
        num_workers=config["evaluation"]["num_workers"],
        pin_memory=True
    )
    
    return dataloaders