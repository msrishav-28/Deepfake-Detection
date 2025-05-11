import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from pathlib import Path

from models.efficientnet import EfficientNetB4Detector
from models.resnet import ResNet50Detector
from models.xception import XceptionDetector
from models.ensemble import DeepfakeEnsemble
from data.datasets import DeepfakeDataset
from utils.metrics import compute_metrics, compute_confusion_matrix
from utils.logger import setup_logger
from utils.checkpointing import load_checkpoint

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []  # Store probability scores
    
    with torch.no_grad():
        pbar = tqdm(dataloader)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probabilities = torch.softmax(outputs, dim=1)
            fake_probs = probabilities[:, 1].cpu().numpy()  # Probability of fake class
            _, predicted = torch.max(outputs, 1)
            
            all_probs.extend(fake_probs)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    eval_loss = running_loss / len(dataloader)
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
    conf_matrix = compute_confusion_matrix(np.array(all_labels), np.array(all_preds))
    
    return eval_loss, metrics, conf_matrix, np.array(all_labels), np.array(all_preds), np.array(all_probs)

def load_model(model_path, device):
    """Load a pre-trained model based on its path"""
    # Determine model type from path
    if "ensemble" in model_path.lower():
        # For ensemble models, need config file to get base models
        config_path = os.path.join(os.path.dirname(model_path), "config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"Ensemble model config not found at {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        base_models = []
        for base_path in config.get("model_paths", []):
            base_model = load_model(base_path, device)
            base_models.append(base_model)
        
        model = DeepfakeEnsemble(base_models, device=device)
    elif "efficientnet" in model_path.lower():
        model = EfficientNetB4Detector(num_classes=2)
    elif "resnet" in model_path.lower():
        model = ResNet50Detector(num_classes=2)
    elif "xception" in model_path.lower():
        model = XceptionDetector(num_classes=2)
    else:
        raise ValueError(f"Cannot determine model type from path: {model_path}")
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    return model

def evaluate_on_dataset(model, dataset_name, data_dir, batch_size, num_workers, device, result_dir, logger):
    """Evaluate a model on a specific dataset"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = DeepfakeDataset(
        root_dir=data_dir,
        dataset_type=dataset_name,
        split="test",
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Evaluating on {dataset_name} dataset with {len(dataset)} samples")
    
    criterion = nn.CrossEntropyLoss()
    loss, metrics, conf_matrix, labels, preds, probs = evaluate_model(model, dataloader, criterion, device)
    
    # Log results
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Loss: {loss:.4f}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"AUC: {metrics['auc']:.4f}")
    logger.info(f"Confusion Matrix:\n{conf_matrix}")
    
    # Save detailed results
    method_result_dir = os.path.join(result_dir, dataset_name)
    os.makedirs(method_result_dir, exist_ok=True)
    
    # Save metrics as JSON
    with open(os.path.join(method_result_dir, "metrics.json"), "w") as f:
        json.dump({
            "loss": loss,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "auc": metrics["auc"],
            "confusion_matrix": conf_matrix.tolist()
        }, f, indent=4)
    
    # Save raw predictions for further analysis
    np.savez(os.path.join(method_result_dir, "predictions.npz"),
             labels=labels, predictions=preds, probabilities=probs)
    
    # Create ROC curve
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["auc"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {dataset_name}')
    plt.legend()
    plt.savefig(os.path.join(method_result_dir, "roc_curve.png"), dpi=300)
    plt.close()
    
    # Create Precision-Recall curve
    precision, recall, _ = precision_recall_curve(labels, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {dataset_name}')
    plt.savefig(os.path.join(method_result_dir, "pr_curve.png"), dpi=300)
    plt.close()
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate deepfake detection models")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing processed face data")
    parser.add_argument("--model_paths", type=str, nargs="+", required=True, 
                        help="Paths to trained model checkpoints")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", 
                        help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--datasets", type=str, nargs="+", default=["faceforensics", "celebdf", "combined"],
                        help="Datasets to evaluate on")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use for evaluation")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(os.path.join(args.output_dir, "evaluation.log"))
    logger.info(f"Evaluating models: {args.model_paths}")
    logger.info(f"On datasets: {args.datasets}")
    
    # Process each model
    for model_path in args.model_paths:
        model_name = os.path.basename(os.path.dirname(model_path))
        logger.info(f"\n=== Evaluating model: {model_name} ===")
        
        # Create result directory for this model
        model_result_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(model_result_dir, exist_ok=True)
        
        try:
            # Load model
            model = load_model(model_path, args.device)
            
            # Evaluate on each dataset
            all_results = {}
            for dataset_name in args.datasets:
                metrics = evaluate_on_dataset(
                    model=model,
                    dataset_name=dataset_name,
                    data_dir=args.data_dir,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    device=args.device,
                    result_dir=model_result_dir,
                    logger=logger
                )
                all_results[dataset_name] = metrics
            
            # Save overall results summary
            with open(os.path.join(model_result_dir, "all_results.json"), "w") as f:
                json.dump(all_results, f, indent=4)
            
        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {str(e)}")
            continue
    
    logger.info("Evaluation completed!")

if __name__ == "__main__":
    main()