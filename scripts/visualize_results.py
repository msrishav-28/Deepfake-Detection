import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_curve, auc
from matplotlib.gridspec import GridSpec

def load_results(results_dir):
    """Load evaluation results from directory"""
    results = {}
    
    # Find all model directories
    model_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    
    for model_name in model_dirs:
        model_path = os.path.join(results_dir, model_name)
        all_results_path = os.path.join(model_path, "all_results.json")
        
        if os.path.exists(all_results_path):
            with open(all_results_path, 'r') as f:
                model_results = json.load(f)
            results[model_name] = model_results
    
    return results

def plot_comparative_metrics(results, output_dir, metrics_to_plot=None):
    """Create comparative bar charts for different metrics"""
    if metrics_to_plot is None:
        metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "auc"]
    
    # Organize data for plotting
    data = []
    for model_name, model_results in results.items():
        for dataset_name, metrics in model_results.items():
            for metric_name in metrics_to_plot:
                if metric_name in metrics:
                    data.append({
                        "Model": model_name,
                        "Dataset": dataset_name,
                        "Metric": metric_name,
                        "Value": metrics[metric_name]
                    })
    
    df = pd.DataFrame(data)
    
    # Plot each metric
    for metric_name in metrics_to_plot:
        plt.figure(figsize=(12, 8))
        metric_data = df[df['Metric'] == metric_name]
        
        sns.barplot(x="Model", y="Value", hue="Dataset", data=metric_data)
        plt.title(f"Comparative {metric_name.replace('_', ' ').title()}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"comparative_{metric_name}.png"), dpi=300)
        plt.close()

def plot_roc_curves(results_dir, output_dir):
    """Create comparative ROC curves"""
    plt.figure(figsize=(10, 8))
    
    # Find all model directories
    model_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    
    for model_name in model_dirs:
        model_path = os.path.join(results_dir, model_name)
        
        # Get dataset directories for this model
        dataset_dirs = [d for d in os.listdir(model_path) 
                        if os.path.isdir(os.path.join(model_path, d)) and d != 'figures']
        
        for dataset_name in dataset_dirs:
            dataset_path = os.path.join(model_path, dataset_name)
            pred_file = os.path.join(dataset_path, "predictions.npz")
            
            if os.path.exists(pred_file):
                data = np.load(pred_file)
                labels = data['labels']
                probs = data['probabilities']
                
                fpr, tpr, _ = roc_curve(labels, probs)
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, lw=2, 
                         label=f'{model_name} - {dataset_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Comparative ROC Curves')
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparative_roc_curves.png"), dpi=300)
    plt.close()

def create_error_analysis_visualizations(results_dir, output_dir):
    """Create visualizations for error analysis"""
    # Find all model directories
    model_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    
    for model_name in model_dirs:
        model_path = os.path.join(results_dir, model_name)
        
        # Get dataset directories for this model
        dataset_dirs = [d for d in os.listdir(model_path) 
                        if os.path.isdir(os.path.join(model_path, d)) and d != 'figures']
        
        for dataset_name in dataset_dirs:
            dataset_path = os.path.join(model_path, dataset_name)
            pred_file = os.path.join(dataset_path, "predictions.npz")
            metrics_file = os.path.join(dataset_path, "metrics.json")
            
            if os.path.exists(pred_file) and os.path.exists(metrics_file):
                # Load data
                data = np.load(pred_file)
                labels = data['labels']
                preds = data['predictions']
                probs = data['probabilities']
                
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                # Create confusion matrix heatmap
                plt.figure(figsize=(8, 6))
                conf_mat = np.array(metrics['confusion_matrix'])
                sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title(f'Confusion Matrix - {model_name} on {dataset_name}')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{model_name}_{dataset_name}_confusion.png"), dpi=300)
                plt.close()
                
                # Plot probability distribution for correct vs incorrect predictions
                plt.figure(figsize=(10, 6))
                correct = preds == labels
                
                plt.hist(probs[correct & (labels == 1)], alpha=0.5, bins=20, 
                         label='Correct (Fake)', color='green')
                plt.hist(probs[~correct & (labels == 1)], alpha=0.5, bins=20, 
                         label='Incorrect (Fake classified as Real)', color='red')
                plt.hist(probs[correct & (labels == 0)], alpha=0.5, bins=20, 
                         label='Correct (Real)', color='blue')
                plt.hist(probs[~correct & (labels == 0)], alpha=0.5, bins=20, 
                         label='Incorrect (Real classified as Fake)', color='orange')
                
                plt.xlabel('Probability of being fake')
                plt.ylabel('Count')
                plt.title(f'Probability Distribution - {model_name} on {dataset_name}')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{model_name}_{dataset_name}_prob_dist.png"), dpi=300)
                plt.close()

def create_performance_summary_table(results, output_dir):
    """Create a summary table of all results"""
    # Create DataFrame for the table
    table_data = []
    
    for model_name, model_results in results.items():
        for dataset_name, metrics in model_results.items():
            row = {
                "Model": model_name,
                "Dataset": dataset_name,
                "Accuracy": metrics.get("accuracy", "-"),
                "Precision": metrics.get("precision", "-"),
                "Recall": metrics.get("recall", "-"),
                "F1 Score": metrics.get("f1_score", "-"),
                "AUC": metrics.get("auc", "-")
            }
            table_data.append(row)
    
    df = pd.DataFrame(table_data)
    
    # Save to CSV
    df.to_csv(os.path.join(output_dir, "performance_summary.csv"), index=False)
    
    # Create a styled HTML table
    styled_df = df.style.format({
        "Accuracy": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1 Score": "{:.4f}",
        "AUC": "{:.4f}"
    }).background_gradient(cmap='Blues', subset=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'])
    
    html = styled_df.to_html()
    with open(os.path.join(output_dir, "performance_summary.html"), "w") as f:
        f.write(html)

def create_comparative_dashboard(results, output_dir):
    """Create a dashboard with multiple plots for comprehensive comparison"""
    model_names = list(results.keys())
    dataset_names = list(set([dataset for model in results.values() for dataset in model.keys()]))
    
    # Create dataframe for easy plotting
    data = []
    for model_name, model_results in results.items():
        for dataset_name, metrics in model_results.items():
            for metric_name, value in metrics.items():
                if metric_name in ["accuracy", "precision", "recall", "f1_score", "auc"]:
                    data.append({
                        "Model": model_name,
                        "Dataset": dataset_name,
                        "Metric": metric_name,
                        "Value": value
                    })
    
    df = pd.DataFrame(data)
    
    # Create a dashboard figure
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 2, figure=fig)
    
    # Plot 1: Bar chart of accuracy across models and datasets
    ax1 = fig.add_subplot(gs[0, 0])
    accuracy_data = df[df['Metric'] == 'accuracy']
    sns.barplot(x="Model", y="Value", hue="Dataset", data=accuracy_data, ax=ax1)
    ax1.set_title("Accuracy Comparison")
    ax1.set_ylim([0.5, 1.0])  # Accuracy usually above 0.5
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 2: F1 score comparison
    ax2 = fig.add_subplot(gs[0, 1])
    f1_data = df[df['Metric'] == 'f1_score']
    sns.barplot(x="Model", y="Value", hue="Dataset", data=f1_data, ax=ax2)
    ax2.set_title("F1 Score Comparison")
    ax2.set_ylim([0.5, 1.0])
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 3: Precision-Recall trade-off
    ax3 = fig.add_subplot(gs[1, 0])
    for model_name in model_names:
        for dataset_name in dataset_names:
            if dataset_name in results[model_name]:
                metrics = results[model_name][dataset_name]
                ax3.scatter(metrics.get("recall", 0), metrics.get("precision", 0), 
                           s=100, label=f"{model_name} - {dataset_name}")
    
    ax3.set_xlim([0.5, 1.0])
    ax3.set_ylim([0.5, 1.0])
    ax3.set_xlabel("Recall")
    ax3.set_ylabel("Precision")
    ax3.set_title("Precision vs Recall")
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 4: AUC comparison
    ax4 = fig.add_subplot(gs[1, 1])
    auc_data = df[df['Metric'] == 'auc']
    sns.barplot(x="Model", y="Value", hue="Dataset", data=auc_data, ax=ax4)
    ax4.set_title("AUC Comparison")
    ax4.set_ylim([0.5, 1.0])
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 5: Performance metrics heatmap - for the best model
    ax5 = fig.add_subplot(gs[2, :])
    
    # Find the best model based on average accuracy
    avg_accuracy = {}
    for model_name in model_names:
        accuracies = [results[model_name][ds]["accuracy"] for ds in results[model_name]]
        avg_accuracy[model_name] = sum(accuracies) / len(accuracies)
    
    best_model = max(avg_accuracy.items(), key=lambda x: x[1])[0]
    
    # Create heatmap data for best model
    heatmap_data = []
    for dataset_name in dataset_names:
        if dataset_name in results[best_model]:
            metrics = results[best_model][dataset_name]
            row = [
                metrics.get("accuracy", 0),
                metrics.get("precision", 0),
                metrics.get("recall", 0),
                metrics.get("f1_score", 0),
                metrics.get("auc", 0)
            ]
            heatmap_data.append(row)
    
    heatmap_df = pd.DataFrame(
        heatmap_data, 
        index=dataset_names,
        columns=["Accuracy", "Precision", "Recall", "F1 Score", "AUC"]
    )
    
    sns.heatmap(heatmap_df, annot=True, fmt=".4f", cmap="YlGnBu", ax=ax5)
    ax5.set_title(f"Performance Metrics Heatmap for Best Model: {best_model}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparative_dashboard.png"), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize deepfake detection evaluation results")
    parser.add_argument("--results_dir", type=str, required=True, 
                        help="Directory containing evaluation results")
    parser.add_argument("--output_dir", type=str, default="./visualization_results", 
                        help="Directory to save visualization results")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    results = load_results(args.results_dir)
    
    if not results:
        print("No evaluation results found in the specified directory.")
        return
    
    # Create visualizations
    print("Creating comparative metrics plots...")
    plot_comparative_metrics(results, args.output_dir)
    
    print("Creating ROC curve comparison...")
    plot_roc_curves(args.results_dir, args.output_dir)
    
    print("Creating error analysis visualizations...")
    create_error_analysis_visualizations(args.results_dir, args.output_dir)
    
    print("Creating performance summary table...")
    create_performance_summary_table(results, args.output_dir)
    
    print("Creating comparative dashboard...")
    create_comparative_dashboard(results, args.output_dir)
    
    print(f"Visualizations completed! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()