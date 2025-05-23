# Deepfake Detection Project

## Overview

This project implements a comprehensive deepfake detection system utilizing various deep learning models, including Vision Transformers (ViT, DeiT, Swin), CNNs (EfficientNet, ResNet, Xception), and advanced fusion techniques. It provides a suite of tools for dataset preprocessing, model training (individual, ensemble, and fusion models), extensive evaluation, results visualization, and API-based deployment for inference. The system is designed to work with popular deepfake datasets like FaceForensics++ and Celeb-DF, offering robust capabilities for detecting manipulated media.

## Features

* **Multiple Model Architectures**: Support for state-of-the-art models like ViT, DeiT, Swin Transformer, EfficientNet, ResNet, and Xception.
* **Advanced Fusion Methods**: Includes ensemble techniques (voting, stacking, weighted averaging) and feature fusion strategies (attention, concatenation, transformer-based fusion).
* **Comprehensive Data Handling**:
    * Preprocessing for datasets like FaceForensics++ and Celeb-DF.
    * Face extraction and frame extraction tools.
    * Data normalization and extensive augmentation (spatial, color, quality degradation).
* **Flexible Training Pipelines**:
    * Scripts for training individual models, ensemble models, and feature fusion models.
    * Modular components for losses (BCE, Focal, Contrastive, Triplet), optimizers (SGD, Adam, AdamW), and learning rate schedulers (Step, MultiStep, Cosine with warmup).
    * Configuration via YAML files for all stages.
* **In-depth Evaluation**:
    * Calculation of various metrics: Accuracy, F1-score, ROC AUC, EER, Precision, Recall, Confusion Matrix.
    * Cross-dataset evaluation support.
    * Visualization tools for ROC curves, Grad-CAM, PCA/t-SNE, and attention maps.
* **Inference and Deployment**:
    * Inference capabilities for single images and videos.
    * Ensemble inference methods (average, voting, weighted, max).
    * FASTAPI-based REST API for easy deployment and integration.
* **Extensibility**: Designed for easy addition of new models, datasets, metrics, fusion techniques, and augmentations.

## Directory Structure

The project follows a modular directory structure:

```
deepfake-detection/
├── LICENSE
├── README.md
├── directory_structure.txt
├── configs/                # Main configuration files (YAML) for various stages
│   ├── data_config.yaml
│   ├── evaluation_config.yaml
│   ├── fusion_config.yaml
│   ├── model_config.yaml
│   └── training_config.yaml
├── data/                   # Data loading, preprocessing, and augmentation modules
│   ├── augmentation/       # Image augmentation techniques (color, quality, spatial)
│   ├── config/             # Dataset-specific configurations
│   ├── dataloaders/        # Dataloader factory
│   ├── datasets/           # Dataset classes (FaceForensics++, Celeb-DF, custom)
│   └── preprocessing/      # Face/frame extraction, normalization
├── evaluation/             # Model evaluation and visualization tools
│   ├── analyze_results.py  # Core evaluation logic and metrics calculation
│   ├── config/             # Evaluation-specific configurations
│   ├── cross_dataset/      # Cross-dataset evaluation scripts
│   ├── metrics/            # Implementation of various evaluation metrics
│   └── visualisation/      # Visualization scripts (ROC, Grad-CAM, attention, etc.)
├── fusion/                 # Ensemble and feature fusion methods
│   ├── config/             # Fusion-specific configurations
│   ├── ensemble/           # Ensemble methods (voting, stacking, weighted averaging)
│   └── feature_fusion/     # Feature fusion techniques (attention, concatenation, transformer)
├── inference/              # Inference scripts and deployment tools
│   ├── config/             # Inference and deployment configurations
│   ├── deployment/         # API deployment using FastAPI
│   │   ├── api.py
│   │   ├── server.py
│   │   └── ...
│   ├── ensemble_inference.py # Script for ensemble-based inference
│   ├── inference.py        # Script for single model inference
│   └── video_inference.py  # Script for video-based inference
├── models/                 # Model definitions
│   ├── base_model.py       # Base model class with common utilities
│   ├── deit/               # DeiT model implementation
│   ├── model_zoo/          # Model factory for creating models
│   ├── swin/               # Swin Transformer model implementation
│   └── vit/                # Vision Transformer (ViT) model implementation
├── notebooks/              # Jupyter notebooks for exploration, demo, and analysis
│   ├── data_exploration.ipynb    # Notebook for exploring datasets
│   ├── demo.ipynb                # Interactive demo of the detection system
│   ├── model_visualization.ipynb # Notebook for visualizing model internals
│   └── results_analysis.ipynb    # Notebook for analyzing evaluation results
├── scripts/                # Helper scripts for various tasks
│   ├── download_faceforensics.py # Script to download FaceForensics++ dataset
│   ├── evaluate_all.py         # Script to evaluate all trained models
│   ├── preprocess_celebdf.py     # Script to preprocess Celeb-DF dataset
│   ├── preprocess_faceforensics.py # Script to preprocess FaceForensics++ dataset
│   ├── train_ensemble.py       # Script to train ensemble models
│   ├── train_individual.py     # Script to train individual models
│   └── visualize_results.py    # Script to generate plots from evaluation results
├── training/               # Training components
│   ├── callbacks/          # Training callbacks (EarlyStopping, ModelCheckpoint)
│   ├── config/             # Training-specific configurations
│   ├── losses/             # Custom loss functions (Focal, Contrastive, Triplet)
│   ├── optimizers/         # Optimizer factory
│   ├── schedulers/         # Learning rate scheduler factory
│   ├── train.py            # Main training script using Trainer class
│   └── trainer.py          # Core Trainer class for managing training loop
└── utils/                  # Utility functions
    ├── config_utils.py     # Utilities for loading/saving YAML configs
    ├── distributed_utils.py # Utilities for distributed training
    ├── file_utils.py       # File system utilities
    ├── logging_utils.py    # Logging setup and AverageMeter
    └── visualization_utils.py # Basic visualization utilities
```
*(Based on content of `directory_structure.txt` and `README.md`)*

## Configuration Files

The project heavily relies on YAML configuration files located in the `configs/` directory and sub-directories like `data/config/`, `evaluation/config/`, etc.. These files control various aspects of the project:

* **`configs/data_config.yaml`**: Defines parameters for dataset loading, preprocessing, and augmentation. Includes paths, split ratios, face detection settings (method, min_face_size, margin), normalization parameters (mean, std, size), and extensive augmentation options (flips, rotations, color adjustments, noise, blur, cutout, etc.).
* **`configs/model_config.yaml`**: Contains templates for different model architectures (ViT, DeiT, Swin and their variants like `vit_small`, `deit_large`, `swin_tiny`) specifying parameters like image size, patch size, embedding dimensions, depth, number of heads, MLP ratio, and dropout rates.
* **`configs/training_config.yaml`**: Manages all settings related to model training experiments. This includes experiment name, seed, output directory, data settings (source, splits, image size), model type and parameters, checkpoint paths for fine-tuning, optimizer settings (type, lr, weight decay), scheduler settings (type, warmup, min_lr), loss function (type and its specific parameters like alpha/gamma for Focal Loss), batch size, number of epochs, mixed precision training, and logging preferences.
* **`configs/evaluation_config.yaml`**: Governs the model evaluation process. Defines experiment details, paths to datasets (FaceForensics++, Celeb-DF, DFDC), image size, model checkpoints to be evaluated along with their parameters, ensemble settings (methods, weights), evaluation batch size, metrics to compute (accuracy, AUC, EER, F1, etc.), datasets to evaluate on, and cross-dataset evaluation parameters.
* **`configs/fusion_config.yaml`**: Specifies parameters for training and evaluating model fusion approaches. Includes experiment settings, data paths, dataset choice, mode (single, ensemble, feature_fusion), parameters for individual models in single mode, ensemble methods (simple_voting, weighted_average, stacking) and model paths for ensemble mode. For feature fusion, it details the method (concat, attention, transformer, bottleneck), fusion dimension, models involved with their feature dimensions and extraction layers, and specific architectural parameters for each fusion type (e.g., hidden layers for MLP in concat, number of heads for attention/transformer).
* **`inference/config/*.yaml`**: These files configure the behavior of inference scripts:
    * `inference_config.yaml`: For single model inference.
    * `ensemble_inference_config.yaml`: For inference with an ensemble of models.
    * `video_inference_config.yaml`: For video-based deepfake detection, including frame sampling, batching, and temporal smoothing.
* **`inference/deployment/config/deployment_config.yaml`**: Configures the FastAPI deployment, including server settings (host, port), API metadata, choice of using a single model or an ensemble for the API backend, device selection, face detector usage, and settings for preprocessing, video processing, output management, explanations (Grad-CAM, attention maps), and caching.
* Other specific config files like `data/config/data_config.yaml` and `evaluation/config/eval_config.yaml` provide context-specific overrides or detailed settings for those modules.

## Setup

1.  **Clone Repository**
    ```bash
    git clone <repository_url>
    cd deepfake-detection
    ```

2.  **Install Dependencies**
    It's recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
    Ensure PyTorch is installed according to your CUDA version if GPU support is desired. Visit the [PyTorch official website](https://pytorch.org/) for instructions.
    Additional dependencies might be required for specific functionalities like `dlib` for some face analysis in notebooks.

3.  **Download Datasets**
    * **FaceForensics++**: Download from the [Official Site](https://faceforensics.org/).
    * **Celeb-DF**: Download from the [GitHub Repo](https://github.com/danmohaha/celeb-deepfakeforensics) or the provided [dataset link](https://github.com/danmohaha/celeb-deepfakeforensics).
    * *(Optional)* DFDC (Deepfake Detection Challenge) dataset.

    Update the paths in the relevant `configs/*.yaml` files (e.g., `configs/data_config.yaml`, `configs/training_config.yaml`) to point to your downloaded dataset locations. For example, update `faceforensics_root`, `celebdf_root`, `dfdc_root`.

4.  **Preprocess Datasets**
    Before training, preprocess the datasets to extract faces (and/or frames).
    * **FaceForensics++**:
        ```bash
        python scripts/preprocess_faceforensics.py --input /path/to/FaceForensics --output /path/to/processed/FaceForensics
        ```
        This script processes original and manipulated sequences (Deepfakes, Face2Face, FaceSwap, NeuralTextures). *(This script uses `data/preprocessing/face_extraction.py`)*
    * **Celeb-DF**:
        ```bash
        python scripts/preprocess_celebdf.py --input /path/to/CelebDF --output /path/to/processed/CelebDF
        ```
        This script processes real and synthetic videos from Celeb-DF. *(This script uses `data/preprocessing/face_extraction.py`)*

    Make sure the `--output` paths here match the data directory specified in your training configuration files. The preprocessing scripts use face detection (e.g., MTCNN) to extract faces from video frames and save them.

## Usage

All primary actions like training, evaluation, and inference are driven by Python scripts in the `scripts/` directory or main scripts within modules (e.g., `training/train.py`, `inference/inference.py`). These scripts typically accept YAML configuration files to manage parameters.

### 1. Training Models

#### a. Training Individual Models
Use `scripts/train_individual.py` to train a single deepfake detection model (e.g., EfficientNet, ResNet, Xception, ViT, DeiT, Swin).
You can specify model type, dataset, and other parameters via command-line arguments or by using a training configuration file.

**Example using command-line arguments:**
```bash
python scripts/train_individual.py \
  --data_dir /path/to/processed/data \
  --model_type efficientnet \
  --dataset faceforensics \
  --output_dir ./trained_models/efficientnet_ff
```

**Example using a configuration file (recommended):**
First, create or modify a configuration file, for example, `configs/training_config.yaml`. Ensure paths and parameters are set correctly.

```yaml
# configs/training_config.yaml (example snippet)
experiment:
  name: "deepfake_detection_vit_celebdf" #
  output_dir: "experiments/vit_celebdf_run1" #
  seed: 42 #
data:
  img_size: 224 #
  celebdf_root: "/path/to/processed/CelebDF" # Ensure this points to preprocessed data
  dataset: "celebdf" #
  # ... other data settings
model:
  type: "vit" # e.g., vit, deit, swin
  # ... model-specific params from configs/model_config.yaml
training:
  batch_size: 32 #
  epochs: 50 #
  optimizer:
    type: "adam" #
    lr: 0.0001 #
  # ... other training settings
logging:
  tensorboard: true #
  save_dir: "logs" # Relative to experiment output_dir
  # ...
```

Then run the training using `training/train.py`:

```bash
python training/train.py \
  --config configs/training_config.yaml \
  --output ./trained_models/vit_celebdf_experiment # Overrides output_dir in config if needed
```
*(The `training/train.py` script uses the `Trainer` class from `training/trainer.py`)*

#### b. Training Ensemble Models

Use `scripts/train_ensemble.py` to train ensemble models by combining pre-trained individual models. This usually involves training a meta-learner or fine-tuning the ensemble.

**Example:**

```bash
python scripts/train_ensemble.py \
  --data_dir /path/to/processed/data \
  --model_paths ./trained_models/model1.pth ./trained_models/model2.pth \
  --dataset combined \
  --output_dir ./trained_models/ensemble_model \
  --freeze_base_models
```
*(The script `train_ensemble.py` initializes an ensemble using multiple model paths)*

Alternatively, refer to `configs/fusion_config.yaml` for ensemble configurations (e.g., `method: "stacking"` or `"weighted_average"`) and adapt `scripts/train_ensemble.py` or a dedicated fusion training script.

#### c. Training Feature Fusion Models

Feature fusion models combine features from multiple base models at an intermediate level.
Configure this using `configs/fusion_config.yaml` by setting `mode: "feature_fusion"` and specifying the fusion method (e.g., `attention`, `concat`, `transformer`) and base models.

```yaml
# configs/fusion_config.yaml (example snippet for feature fusion)
experiment:
  name: "deepfake_detection_feature_fusion" #
  output_dir: "fusion_results/attention_vit_deit" #
mode: "feature_fusion" #
dataset: "celebdf" #
data:
  celebdf_root: "/path/to/processed/CelebDF" #
  # ...
fusion:
  method: "attention" # Options: concat, attention, transformer, bottleneck
  fusion_dim: 512 #
  models:
    - type: "vit" #
      checkpoint: "/path/to/checkpoints/vit_best.pth" #
      feature_dim: 768 #
      feature_layer: "cls_token" # Or other appropriate layer
    - type: "deit" #
      checkpoint: "/path/to/checkpoints/deit_best.pth" #
      feature_dim: 768 #
      feature_layer: "cls_token" #
  architecture:
    attention: # Parameters specific to attention fusion
      num_heads: 8 #
      dropout: 0.1 #
# ... training settings for the fusion model
```
*(Ensure `configs/fusion_config.yaml` has `mode: "feature_fusion"` and other relevant parameters set)*

A dedicated training script for fusion models, similar to `training/train.py` but using `fusion_config.yaml`, would be used.

### 2. Evaluating Models

Use `scripts/evaluate_all.py` to evaluate trained models on specified datasets. This script calculates various metrics and can generate visualizations.

**Example:**

```bash
python scripts/evaluate_all.py \
  --data_dir /path/to/processed/data \
  --model_paths ./trained_models/model1.pth ./trained_models/another_model.pth \
  --datasets faceforensics celebdf \
  --output_dir ./evaluation_results
```
*(This script loads models and evaluates them, saving metrics and plots)*

This script uses `evaluation/analyze_results.py` internally. Configuration for evaluation can also be managed via `configs/evaluation_config.yaml`, which specifies models, datasets, metrics, and visualization options.

### 3. Visualizing Evaluation Results

Use `scripts/visualize_results.py` to generate plots and visualizations from the output of `evaluate_all.py`.

**Example:**

```bash
python scripts/visualize_results.py \
  --results_dir ./evaluation_results \
  --output_dir ./visualization_output
```
*(This script generates comparative bar charts, ROC curves, and other visualizations)*

### 4. Running Inference

The `inference/` directory contains tools for performing deepfake detection on new images or videos.

#### a. Single Image Inference

Use `inference/inference.py`. This script loads a model and predicts on a single image.
It can be configured using `inference/config/inference_config.yaml`.

```bash
python inference/inference.py \
  --config inference/config/inference_config.yaml \
  --image /path/to/your/image.jpg \
  --output /path/to/output_visualizations # Optional, for explanations
```

#### b. Ensemble Inference on Image

Use `inference/ensemble_inference.py`. This script loads multiple models and combines their predictions.
Configure using `inference/config/ensemble_inference_config.yaml`.

```bash
python inference/ensemble_inference.py \
  --config inference/config/ensemble_inference_config.yaml \
  --image /path/to/your/image.jpg \
  --output /path/to/output_visualizations # Optional
```

#### c. Video Inference

Use `inference/video_inference.py`. This script processes videos frame by frame, applies detection, and can aggregate results.
Configure using `inference/config/video_inference_config.yaml`.

```bash
python inference/video_inference.py \
  --config inference/config/video_inference_config.yaml \
  --video /path/to/your/video.mp4 \
  --output /path/to/annotated_video.mp4 \
  --analysis /path/to/analysis_video.mp4 # Optional, for visualization of scores over time
```

### 5. API Deployment

The project includes a FastAPI application for deploying the deepfake detection system as a REST API.

1.  **Configure Deployment**: Modify `inference/deployment/config/deployment_config.yaml` to set server options, model paths (single or ensemble), and other API behaviors.

2.  **Run Server**:

    ```bash
    python inference/deployment/server.py --config inference/deployment/config/deployment_config.yaml
    ```
    *(This uses `inference/deployment/api.py` to create the API and `server.py` to run it)*

    The API server will start (default: `http://0.0.0.0:8000`). Endpoints will be available for image and video detection, including options for explanations. See `inference/deployment/api.py` for endpoint details.

    Key endpoints typically include:

      * `POST /detect/image`: Upload an image for detection.
      * `POST /detect/video`: Upload a video for detection (runs as a background task).
      * `GET /detect/video/status/{task_id}`: Check video processing status.
      * `GET /detect/video/result/{video_id}`: Retrieve annotated video.

## In-Depth Evaluation (API Example from original README)

The `evaluation/analyze_results.py` module provides a `ModelEvaluator` class for programmatic evaluation.

```python
from evaluation.analyze_results import ModelEvaluator #
from torch.utils.data import DataLoader #
import torch # Assuming torch is imported

# Assume:
# model = Your trained PyTorch model (loaded and set to device)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# test_loader = DataLoader for your test dataset

# Example (ensure model and test_loader are defined)
# model = create_model(model_type='vit', **vit_params) # Example
# model.load_state_dict(torch.load('path/to/your/model.pth')['model']) #
# model.to(device).eval() #
# test_dataset = ... # Your test dataset instance
# test_loader = DataLoader(test_dataset, batch_size=64)


evaluator = ModelEvaluator(model, device=device) #
metrics = evaluator.evaluate(test_loader) # Pass your DataLoader
print("Calculated Metrics:", metrics) #

report = evaluator.generate_classification_report() # Generates sklearn classification report
print("Classification Report:\n", report) #

# Visualize results (e.g., ROC curve, confusion matrix)
# Ensure output_dir exists or is handled
figures = evaluator.visualize_results(output_dir="./evaluation_figures") #
if "roc_curve" in figures and figures["roc_curve"] is not None: #
    figures["roc_curve"].savefig("roc_curve.png") #
if "confusion_matrix" in figures and figures["confusion_matrix"] is not None: #
    figures["confusion_matrix"].savefig("confusion_matrix.png") #
```

## Extensibility

The project is designed to be extensible in several ways:

  * **Adding New Models**:
    1.  Implement your model class, inheriting from `models.base_model.BaseModel` if desired, in a new file under `models/your_model_type/`.
    2.  Ensure it has `forward(self, x)` and optionally `extract_features(self, x)` methods.
    3.  Register your model in `models/model_zoo/model_factory.py` by adding a new condition to the `create_model` function.
    4.  Add a configuration template for your model in `configs/model_config.yaml`.
  * **Adding Custom Datasets**:
    1.  Create a new dataset class in `data/datasets/` inheriting from `torch.utils.data.Dataset`.
    2.  Implement `__init__`, `__len__`, and `__getitem__`.
    3.  Update `data/dataloaders/dataloader_factory.py` to include your new dataset.
    4.  Add relevant configuration options to `configs/data_config.yaml`.
  * **Adding New Evaluation Metrics**:
    1.  Implement the metric calculation in a new file under `evaluation/metrics/`.
    2.  Integrate its calculation into `evaluation/analyze_results.py` or the `evaluate_all.py` script.
    3.  Add the metric name to the `metrics` list in `configs/evaluation_config.yaml`.
  * **Adding New Fusion Techniques**:
    1.  For ensemble methods, add a new class in `fusion/ensemble/` (e.g., similar to `SimpleVoting`).
    2.  For feature fusion, add a new class in `fusion/feature_fusion/` (e.g., similar to `AttentionFusion`).
    3.  Update `fusion/__init__.py` and potentially the fusion training scripts/configuration (`configs/fusion_config.yaml`).
  * **Adding New Augmentations**:
    1.  Implement the augmentation function, possibly using libraries like Albumentations.
    2.  Add it to one of the files in `data/augmentation/` (e.g., `spatial_augmentations.py`).
    3.  Expose it through `data/augmentation/__init__.py` and add a configuration option in `configs/data_config.yaml`.

## Notebooks

The `notebooks/` directory contains Jupyter notebooks for various purposes:

  * **`data_exploration.ipynb`**: Tools and examples for exploring and understanding the datasets (e.g., class distributions, sample visualization, image properties).
  * **`demo.ipynb`**: An interactive demonstration of the deepfake detection system, allowing users to test models on their own images/videos and visualize results.
  * **`model_visualization.ipynb`**: Notebook focused on visualizing model internals, such as attention maps and Grad-CAM, to understand model behavior.
  * **`results_analysis.ipynb`**: Scripts and visualizations for in-depth analysis of evaluation results, comparing models, and identifying error patterns.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Authors

**M S Rishav Subhin © 2025**
**Swarnajit Roy © 2025**
