# Deepfake Detection Project — README

## Overview

This project implements a comprehensive deepfake detection system using various deep learning models and fusion techniques. It includes a suite of tools for dataset preprocessing, model training (individual and ensemble), in-depth performance evaluation, results visualization, and API-based deployment for inference. The project is designed to work with popular deepfake datasets like FaceForensics++ and Celeb-DF, and is extensible for new models, datasets, and techniques.

## Features

  * **Dataset Handling**: Supports FaceForensics++, Celeb-DF, and custom combined datasets.
  * **Preprocessing**: Includes face extraction using MTCNN/RetinaFace, frame extraction from videos, and image normalization.
  * **Data Augmentation**: Comprehensive augmentation techniques including spatial, color, and quality-based degradations.
  * **Deep Learning Models**:
      * Vision Transformers (ViT, DeiT, Swin Transformer).
      * CNNs (EfficientNet, ResNet, Xception - placeholders in original README, further details can be added if model files are provided).
      * Model factory for easy integration of new architectures.
  * **Training**:
      * Individual and ensemble model training scripts.
      * Modular training components: configurable optimizers, schedulers, and loss functions (BCE, Focal Loss, Contrastive, Triplet).
      * Callbacks for early stopping and model checkpointing.
      * TensorBoard logging for monitoring training progress.
  * **Fusion Techniques**:
      * Ensemble methods: Simple Voting, Weighted Averaging, Stacking.
      * Feature fusion methods: Concatenation, Attention-based fusion, Transformer-based fusion.
  * **Evaluation**:
      * Comprehensive evaluation scripts and tools.
      * Metrics: Accuracy, Precision, Recall, F1-score, AUC, EER, Confusion Matrix.
      * Cross-dataset evaluation support.
  * **Visualization**:
      * Results plotting: ROC curves, Precision-Recall curves, metric comparison plots.
      * Model interpretability: Grad-CAM, Attention Maps, t-SNE/PCA feature space visualization.
  * **Inference**:
      * Scripts for image and video inference.
      * Ensemble inference support.
      * API for deployment using FastAPI.
  * **Configuration**: Highly configurable using YAML files for datasets, models, training, evaluation, fusion, and inference/deployment.
  * **Extensibility**: Designed for easy addition of new models, datasets, metrics, fusion techniques, and augmentations.

## Directory Structure

The project follows a modular directory structure:

```
Deepfake-Detection/
├── LICENSE
├── README.md
├── requirements.txt
├── configs/                # YAML configuration files for various stages
│   ├── data_config.yaml
│   ├── evaluation_config.yaml
│   ├── fusion_config.yaml
│   ├── model_config.yaml
│   └── training_config.yaml
├── data/                   # Data loading, preprocessing, augmentation
│   ├── augmentation/       # Image augmentation techniques
│   ├── config/             # Data-specific configurations
│   ├── dataloaders/        # Dataloader implementations
│   ├── datasets/           # Dataset class definitions
│   └── preprocessing/      # Face extraction, frame extraction, normalization
├── evaluation/             # Model evaluation and visualization
│   ├── config/             # Evaluation-specific configurations
│   ├── cross_dataset/      # Cross-dataset evaluation tools
│   ├── metrics/            # Evaluation metric calculations
│   └── visualisation/      # Visualization scripts (ROC, Grad-CAM, etc.)
├── fusion/                 # Ensemble and feature fusion methods
│   ├── config/             # Fusion-specific configurations
│   ├── ensemble/           # Ensemble learning techniques
│   └── feature_fusion/     # Feature-level fusion techniques
├── inference/              # Inference scripts and deployment tools
│   ├── config/             # Inference and deployment configurations
│   └── deployment/         # API and server for deployment
├── models/                 # Model definitions
│   ├── deit/               # DeiT model components
│   ├── model_zoo/          # Model factory
│   ├── swin/               # Swin Transformer components
│   └── vit/                # Vision Transformer components
├── notebooks/              # Jupyter notebooks for exploration and demos
├── scripts/                # Main scripts for preprocessing, training, evaluation
├── training/               # Training components (trainer, losses, optimizers, etc.)
│   ├── callbacks/          # Training callbacks (early stopping, checkpointing)
│   ├── config/             # Training-specific configurations
│   ├── losses/             # Custom loss functions
│   ├── optimizers/         # Optimizer factory
│   └── schedulers/         # Learning rate scheduler factory
└── utils/                  # Utility functions (config loading, logging, etc.)
```

*(Based on content of `directory_structure.txt` and `README.md`)*

## Key Modules

### `configs/`

This directory holds YAML configuration files that control various aspects of the project.

  * `data_config.yaml`: Defines paths to datasets, splits, preprocessing parameters (face detection, normalization), and augmentation settings.
  * `evaluation_config.yaml`: Configures model evaluation, including experiment settings, data paths, model checkpoints, metrics, and visualization options. The evaluation-specific version is `evaluation/config/eval_config.yaml`.
  * `fusion_config.yaml`: Specifies settings for model fusion experiments, including data, mode (ensemble, feature\_fusion), models to fuse, fusion methods, and training/evaluation parameters for fused models.
  * `model_config.yaml`: Contains configuration templates for different deepfake detection models like ViT, DeiT, and Swin Transformer, including their variants (small, large, tiny) and hyperparameters. Specific model configs are also in `models/<model_type>/config/`.
  * `training_config.yaml`: Defines parameters for model training experiments, such as dataset choice, model type, hyperparameters, optimizer, scheduler, loss function, augmentation, and logging settings. The training-specific version is `training/config/training_config.yaml`.
  * `inference/config/`: Contains configurations for different inference modes:
      * `ensemble_inference_config.yaml`: For running inference with an ensemble of models.
      * `inference_config.yaml`: For running inference with a single model.
      * `video_inference_config.yaml`: For processing videos and performing deepfake detection on them.
  * `inference/deployment/config/deployment_config.yaml`: Configures the FastAPI server for deployment, including server settings, API details, model selection (single/ensemble), and paths.

### `data/`

Handles all data-related operations.

  * **`augmentation/`**: Implements various data augmentation techniques.
      * `color_augmentations.py`: Provides color-based augmentations like brightness/contrast adjustments, hue/saturation shifts, and RGB shifts using Albumentations.
      * `quality_degradation.py`: Simulates real-world quality degradations including image compression, Gaussian blur, Gaussian noise, and motion blur.
      * `spatial_augmentations.py`: Implements spatial augmentations such as horizontal flip, and shift/scale/rotate operations.
  * **`config/`**: Contains `data_config.yaml` for dataset paths, preprocessing, and augmentation settings, similar to the main `configs/data_config.yaml`.
  * **`dataloaders/`**: Manages the creation of PyTorch DataLoaders.
      * `dataloader_factory.py`: A factory function `create_dataloaders` to instantiate training, validation, and test dataloaders based on the provided configuration, supporting FaceForensics++, Celeb-DF, and combined datasets.
  * **`datasets/`**: Defines PyTorch Dataset classes for different deepfake datasets.
      * `celebdf.py`: Implements `CelebDFDataset` for the Celeb-DF dataset, handling data loading and splitting.
      * `custom_dataset.py`: Provides `DeepfakeDataset` which can combine multiple individual datasets into one, useful for training on mixed data.
      * `faceforensics.py`: Implements `FaceForensicsDataset` for the FaceForensics++ dataset, allowing selection of specific manipulation methods.
  * **`preprocessing/`**: Contains scripts for preparing raw data.
      * `face_extraction.py`: Includes functions to detect and extract faces from images and video frames using MTCNN. It allows setting margins and minimum face size.
      * `frame_extraction.py`: Provides functionality to extract frames from videos at a specified sample rate.
      * `normalization.py`: Contains functions for normalizing face images to a target size and applying standard ImageNet normalization. Also provides basic train/test PyTorch transforms.

### `evaluation/`

Contains modules for evaluating model performance and analyzing results.

  * `analyze_results.py`: Core script with `ModelEvaluator` class for evaluating models, calculating various metrics (accuracy, precision, recall, F1, AUC), generating classification reports, and visualizing results like ROC curves and confusion matrices. Also includes functions to compare multiple models.
  * **`config/`**:
      * `eval_config.yaml`: Configuration for model evaluation, specifying data paths, image size, metrics, and model checkpoint details.
  * **`cross_dataset/`**: Supports evaluating models on datasets different from their training set.
      * `cross_evaluation.py`: Implements cross-dataset evaluation logic, calculating metrics and generalization gaps between source and target datasets.
  * **`metrics/`**: Defines functions for calculating various performance metrics.
      * `classification_metrics.py`: Calculates standard classification metrics like accuracy, precision, recall, F1-score, AUC, and EER.
      * `confusion_matrix.py`: Computes and normalizes confusion matrices.
      * `roc_metrics.py`: Calculates ROC curve points (FPR, TPR), AUC, and EER.
  * **`visualisation/`** (also referred to as `visualization/`): Tools for creating visualizations to understand model behavior and results.
      * `attention_maps.py`: Functions to extract and visualize attention maps from transformer models.
      * `feature_visualisation.py`: Visualizes high-dimensional feature spaces using t-SNE or PCA.
      * `grad_cam.py`: Implements Grad-CAM for visualizing important image regions for model decisions.
      * `results_plots.py`: Functions to plot ROC curves, confusion matrices, and training/validation metric progression over epochs.

### `fusion/`

Implements methods for combining multiple models or their features.

  * **`config/`**:
      * `fusion_config.yaml`: Configuration for model fusion experiments, detailing data settings, fusion mode (ensemble or feature fusion), specific models, fusion methods (e.g., concatenation, attention), and training/evaluation parameters for the fused model.
  * **`ensemble/`**: Provides various ensemble learning techniques.
      * `simple_voting.py`: Implements an ensemble where predictions are made by majority voting among base models.
      * `stacking.py`: Implements stacking, where a meta-model learns to combine predictions from base models.
      * `weighted_average.py`: Implements an ensemble that combines model predictions using a weighted average.
  * **`feature_fusion/`**: Implements techniques for fusing features extracted from multiple models.
      * `attention_fusion.py`: Fuses features using a cross-model attention mechanism.
      * `concat_fusion.py`: Fuses features by simple concatenation followed by a fusion network.
      * `transformer_fusion.py`: Uses a transformer architecture to fuse features from multiple models, incorporating a special fusion token and positional embeddings.

### `inference/`

Handles model inference for deepfake detection on new images or videos.

  * `inference.py`: Defines `DeepfakeDetector` class for loading a single model and performing predictions. Includes preprocessing and explanation generation (Grad-CAM, attention maps). Also includes `load_detector` to load a detector from a config file.
  * `ensemble_inference.py`: Defines `EnsembleDetector` class for combining predictions from multiple `DeepfakeDetector` instances using methods like averaging, voting, or weighted averaging.
  * `video_inference.py`: Defines `VideoDetector` for processing videos, sampling frames, and applying a detector (single or ensemble) to each frame. It can generate annotated videos and analysis summaries.
  * **`config/`**:
      * `inference_config.yaml`: Configuration for single model inference, including model path, device, and preprocessing/explanation settings.
      * `ensemble_inference_config.yaml`: Configuration for ensemble model inference, specifying multiple model paths, ensemble method, and weights.
      * `video_inference_config.yaml`: Configuration for video inference, including model/ensemble choice, video processing parameters (sample rate, batch size), and output settings.
  * **`deployment/`**: Contains modules for deploying the deepfake detection system as an API.
      * `api.py`: Defines the FastAPI application, including routes for image and video detection, explanation generation, and status checking for video processing tasks.
      * `server.py`: Main script to run the FastAPI server using Uvicorn.
      * `utils.py`: Utility functions for deployment, such as loading the deployment configuration.
      * **`config/`**:
          * `deployment_config.yaml`: Configuration for the API deployment, including server settings, API metadata, model selection (single/ensemble), and paths for models and outputs.

### `models/`

Contains definitions for the deep learning models used in the project.

  * `base_model.py`: Defines a `BaseModel` class with common methods for saving and loading checkpoints.
  * **`model_zoo/`**:
      * `model_factory.py`: A factory function `create_model` to instantiate models (ViT, DeiT, Swin) based on type and parameters, simplifying model creation.
  * **`vit/`**: Implementation of the Vision Transformer.
      * `model.py`: Defines the `ViT` model architecture, including patch embedding, transformer blocks, and classification head.
      * `blocks.py`: Contains `Attention` and `MLP` classes used within the `TransformerBlock` for ViT.
      * `embedding.py`: Defines `PatchEmbedding` class for converting images into sequences of flattened patches.
      * `config/vit_config.yaml`: Example configuration for a ViT model.
  * **`deit/`**: Implementation of the Data-efficient Image Transformer.
      * `model.py`: Defines the `DeiT` model, which extends ViT with a distillation token and a distillation head.
      * `distillation.py`: Defines the `DistillationToken` used in DeiT.
      * `attention.py`: Notes that DeiT uses the same attention mechanism as ViT.
      * `config/deit_config.yaml`: Example configuration for a DeiT model.
  * **`swin/`**: Implementation of the Swin Transformer.
      * `model.py`: Defines the `SwinTransformer` architecture, including patch embedding, Swin Transformer blocks arranged in stages, and patch merging layers.
      * `blocks.py`: Contains `WindowAttention`, `SwinTransformerBlock`, `MLP`, and `BasicLayer` which are fundamental components of the Swin Transformer architecture.
      * `patch_merging.py`: Defines the `PatchMerging` layer used in Swin Transformer for hierarchical feature representation by downsampling feature maps (content appears to be merged with `blocks.py` in provided files).
      * `config/swin_config.yaml`: Example configuration for a Swin Transformer model.

### `notebooks/`

Jupyter notebooks for various project tasks:

  * `data_exploration.ipynb`: For exploring and understanding the datasets (FaceForensics++, Celeb-DF), including statistics, visualizations, class distributions, and image properties.
  * `demo.ipynb`: Provides an interactive demonstration of the deepfake detection system, allowing users to load models, analyze images/videos, and visualize results and explanations.
  * `model_visualization.ipynb`: Focuses on visualizing model architectures, attention maps (for transformers), Grad-CAM analysis, and feature space representations to interpret model behavior.
  * `results_analysis.ipynb`: Tools for analyzing evaluation results, including comparative analysis of models, error analysis, cross-dataset generalization, ROC/PR curves, and assessment of ensembling/fusion performance.

### `scripts/`

Contains command-line scripts for key project workflows.

  * `download_faceforensics.py`: Script to download specified parts of the FaceForensics++ dataset (e.g., specific manipulation methods, compression levels).
  * `train_individual.py`: Trains individual deepfake detection models (e.g., EfficientNet, ResNet, Xception, ViT, DeiT, Swin) based on a configuration.
  * `train_ensemble.py`: Trains ensemble models by combining pre-trained individual models.
  * `evaluate_all.py`: Evaluates trained models (single or ensemble) on specified datasets and saves metrics and visualizations.
  * `visualize_results.py`: Generates plots and visualizations from saved evaluation results, including comparative metrics, ROC curves, and confusion matrices.
  * `preprocess_celebdf.py`: Preprocesses the Celeb-DF dataset by extracting faces from videos.
  * `preprocess_faceforensics.py`: Preprocesses the FaceForensics++ dataset by extracting faces from videos of original and manipulated sequences.

### `training/`

Components related to the model training process.

  * `trainer.py`: Defines the `Trainer` class, which encapsulates the training and validation loops, optimizer and scheduler management, loss calculation, metric logging (including TensorBoard), checkpointing, and early stopping logic.
  * `train.py`: Script to initiate model training using a specified configuration file and the `Trainer` class.
  * **`callbacks/`**: Implements callback functions for use during training.
      * `early_stopping.py`: Provides `EarlyStopping` callback to halt training if a monitored metric (e.g., validation loss) stops improving for a defined number of epochs.
      * `model_checkpoint.py`: Implements `ModelCheckpoint` callback to save model weights (best performing or all epochs) during training based on a monitored metric.
  * **`config/`**:
      * `training_config.yaml`: YAML file specifying detailed training parameters such as dataset paths, model architecture and parameters, batch size, epochs, optimizer settings, learning rate scheduler, loss function, and logging options.
  * **`losses/`**: Defines custom and standard loss functions.
      * `focal_loss.py`: Implements `BinaryFocalLoss` to address class imbalance by down-weighting well-classified examples.
      * `contrastive_loss.py`: Implements `ContrastiveLoss`, often used for representation learning by pulling similar samples together and pushing dissimilar ones apart.
      * `triplet_loss.py`: Implements `TripletLoss`, which aims to learn embeddings such that an anchor sample is closer to a positive sample (same class) than to a negative sample (different class) by a certain margin.
  * **`optimizers/`**:
      * `optimizer_factory.py`: A factory function `create_optimizer` to instantiate optimizers (e.g., SGD, Adam, AdamW) based on configuration.
  * **`schedulers/`**:
      * `scheduler_factory.py`: A factory function `create_scheduler` to instantiate learning rate schedulers (e.g., StepLR, MultiStepLR, CosineAnnealingLR with warmup).
  * `tensorboard_logger.py`: Defines `TensorBoardLogger` for logging training metrics (loss, accuracy, etc.) to TensorBoard for real-time visualization.

### `utils/`

A collection of utility functions used across the project.

  * `config_utils.py`: Functions for loading, saving, and merging YAML configuration files.
  * `logging_utils.py`: Provides `setup_logger` for configuring loggers and `AverageMeter` class for tracking running averages of metrics.
  * `visualization_utils.py`: Utility functions for image and mask visualization, such as converting tensors to images and plotting multiple images.
  * `file_utils.py`: Helper functions for file system operations like ensuring directory existence, copying files, and saving/loading JSON and pickle files.
  * `distributed_utils.py`: Utilities for setting up and managing distributed training environments.

## Setup

1.  **Clone Repository**

    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install Dependencies**
    It is recommended to create a virtual environment (e.g., using `conda` or `venv`).

    ```bash
    pip install -r requirements.txt
    ```

    Ensure PyTorch is installed compatible with your CUDA version if using GPU.

3.  **Download Datasets**

      * **FaceForensics++**: Download from the [Official Site](https://www.google.com/url?sa=E&source=gmail&q=https://faceforensics.org/).
      * **Celeb-DF**: Download from the [GitHub Repo](https://github.com/danmohaha/celeb-deepfakeforensics) or other official sources.
      * Place the datasets in a designated directory (e.g., `/path/to/datasets/`).

4.  **Update Configuration Files**

      * Modify the YAML configuration files in the `configs/` directory (and sub-directories like `data/config/`, `training/config/` etc.) to reflect the correct paths to your datasets, desired model parameters, and experiment settings.
      * Key paths to update are typically `root`, `faceforensics_root`, `celebdf_root` in `configs/data_config.yaml`, `configs/training_config.yaml`, `configs/evaluation_config.yaml`, etc.

5.  **Preprocess Datasets**
    Extract faces from the downloaded video datasets.

      * **FaceForensics++**:
        ```bash
        python scripts/preprocess_faceforensics.py \
            --input /path/to/FaceForensics \
            --output /path/to/processed/FaceForensics
        ```
        *(This script uses `data/preprocessing/face_extraction.py`)*
      * **Celeb-DF**:
        ```bash
        python scripts/preprocess_celebdf.py \
            --input /path/to/CelebDF \
            --output /path/to/processed/CelebDF
        ```
        *(This script uses `data/preprocessing/face_extraction.py`)*

## Configuration System

The project heavily relies on YAML configuration files for managing experiments and parameters.

  * **Main Configurations (`configs/`)**:
      * `data_config.yaml`: General settings for datasets, preprocessing, and augmentations.
      * `model_config.yaml`: Base configurations and parameters for different model architectures (ViT, DeiT, Swin, and their variants).
      * `training_config.yaml`: Defines a complete training experiment, including data source, model choice, hyperparameters, optimizer, scheduler, loss, etc.
      * `evaluation_config.yaml`: Settings for evaluating trained models, specifying datasets, metrics, and model checkpoints.
      * `fusion_config.yaml`: Parameters for training and evaluating model fusion approaches (ensemble or feature fusion).
  * **Module-Specific Configurations**:
      * `data/config/data_config.yaml`: Can override or specify data-related settings.
      * `evaluation/config/eval_config.yaml`: Focused configuration for a specific evaluation run.
      * `training/config/training_config.yaml`: Can be used for specific training setups, often referenced by the main `training/train.py` script.
      * `fusion/config/fusion_config.yaml`: Specifics for fusion methods.
      * `inference/config/*.yaml`: Configurations for running inference with single models, ensembles, or on videos.
      * `inference/deployment/config/deployment_config.yaml`: Settings for the FastAPI deployment.
      * `models/<model_type>/config/<model_type>_config.yaml` (e.g., `models/vit/config/vit_config.yaml`): Default parameters for a specific model architecture.

Utilities in `utils/config_utils.py` are used to load and manage these configurations.

## Usage

### Training

#### Train Individual Models

Use the `scripts/train_individual.py` script or `training/train.py` with a configuration file.

**Using `scripts/train_individual.py` (legacy/direct script):**

```bash
python scripts/train_individual.py \
  --data_dir /path/to/processed_data \
  --model_type efficientnet \
  --dataset faceforensics \
  --output_dir ./trained_models \
  --epochs 30 \
  --batch_size 32 \
  --learning_rate 0.0001
```

*(This script takes numerous arguments to define the training process)*

**Using `training/train.py` (config-driven):**
Create or modify a YAML configuration file (e.g., `configs/training_config.yaml` or a copy).
Example `my_training_config.yaml`:

```yaml
# my_training_config.yaml
experiment:
  name: "vit_celebdf_custom_run"
  # ... other experiment settings

data:
  img_size: 224
  celebdf_root: "/path/to/processed/CelebDF" # IMPORTANT: Update this path
  dataset: "celebdf"
  # ... other data settings

model:
  type: "vit"
  params:
    img_size: 224
    # ... other ViT parameters from model_config.yaml
  # ckpt_path: null # for training from scratch

training:
  batch_size: 32
  epochs: 50
  optimizer:
    type: "adamw"
    lr: 0.0001
  # ... other training settings from training_config.yaml

logging:
  tensorboard: True
  save_dir: "logs/my_experiment" # Tensorboard logs
  # ... other logging settings
```

Then run:

```bash
python training/train.py \
  --config path/to/my_training_config.yaml \
  --output ./trained_models/my_vit_celebdf_experiment
```

*(The `training/train.py` script uses the `Trainer` class from `training/trainer.py`)*

#### Train Ensemble Models

Use the `scripts/train_ensemble.py` script. This typically involves fine-tuning a meta-learner on top of frozen base models or end-to-end fine-tuning of the ensemble.

```bash
python scripts/train_ensemble.py \
  --data_dir /path/to/processed_data \
  --model_paths ./trained_models/model1_best.pth ./trained_models/model2_best.pth \
  --dataset combined \
  --output_dir ./trained_models/ensembles \
  --epochs 15 \
  --learning_rate 0.00005 \
  --freeze_base_models
```

*(The script `train_ensemble.py` initializes an ensemble using multiple model paths)*

#### Train Fusion Models (Feature Fusion)

To train feature fusion models, you would typically adapt the `training/train.py` script or create a new one that uses a fusion model defined in `fusion/feature_fusion/` and configured via `configs/fusion_config.yaml`. The `fusion_config.yaml` allows specifying `mode: "feature_fusion"`, the base models, their checkpoints, feature dimensions, and the fusion architecture (e.g., attention, concat, transformer).

Example command structure (conceptual, actual script might need to be `training/train.py` adapted for fusion):

```bash
python training/train.py \
  --config configs/fusion_config.yaml \
  --output ./trained_models/my_feature_fusion_model
```

*(Ensure `configs/fusion_config.yaml` has `mode: "feature_fusion"` and other relevant parameters set)*

### Evaluation

#### Evaluate All Trained Models

Use the `scripts/evaluate_all.py` script to evaluate one or more trained models on specified datasets.

```bash
python scripts/evaluate_all.py \
  --data_dir /path/to/processed_data \
  --model_paths ./trained_models/my_model/best.pth ./trained_models/my_ensemble/best.pth \
  --datasets faceforensics celebdf \
  --output_dir ./evaluation_results
```

*(This script loads models and evaluates them, saving metrics and plots)*

It uses the `ModelEvaluator` from `evaluation/analyze_results.py` and can perform cross-dataset evaluation using logic from `evaluation/cross_dataset/cross_evaluation.py`. Results, including metrics and raw predictions, are saved per model and dataset.

#### Visualize Evaluation Results

Use `scripts/visualize_results.py` to generate comparative plots from the output of `evaluate_all.py`.

```bash
python scripts/visualize_results.py \
  --results_dir ./evaluation_results \
  --output_dir ./visualization_output
```

*(This script generates comparative bar charts, ROC curves, and other visualizations)*

### Inference

#### Image Inference

Use the `inference/inference.py` script with a configuration file.
Create/modify `configs/inference_config.yaml`:

```yaml
# inference_config.yaml
device: "cuda"
face_detector: true
model:
  type: "vit" # or deit, swin
  checkpoint: "/path/to/your/trained_models/my_vit_model/best.pth"
  params: # Ensure these match the trained model's config
    img_size: 224
    # ... other model params
# ... other settings from inference_config.yaml
```

Run:

```bash
python inference/inference.py \
  --config configs/inference_config.yaml \
  --image /path/to/input_image.jpg \
  --explain # Optional: to generate Grad-CAM/attention maps
  # --output /path/to/save_explanation.png # Optional
```

*(The `inference.py` script defines `DeepfakeDetector` for single model inference)*

#### Ensemble Image Inference

Use `inference/ensemble_inference.py` with a configuration file.
Create/modify `configs/ensemble_inference_config.yaml`:

```yaml
# ensemble_inference_config.yaml
device: "cuda"
face_detector: true
ensemble:
  method: "weighted" # or average, voting, max
  weights: [0.4, 0.3, 0.3] # if method is weighted
  models:
    - type: "vit"
      checkpoint: "/path/to/model1.pth"
      params: { ... }
    - type: "deit"
      checkpoint: "/path/to/model2.pth"
      params: { ... }
    # ... more models
# ... other settings from ensemble_inference_config.yaml
```

Run:

```bash
python inference/ensemble_inference.py \
  --config configs/ensemble_inference_config.yaml \
  --image /path/to/input_image.jpg \
  --explain # Optional
  # --model_index 0 # Optional: to explain a specific model in the ensemble
  # --output /path/to/save_explanation.png # Optional
```

*(The `ensemble_inference.py` script defines `EnsembleDetector`)*

#### Video Inference

Use `inference/video_inference.py` with a configuration file.
Create/modify `configs/video_inference_config.yaml`:

```yaml
# video_inference_config.yaml
device: "cuda"
face_detector: true
# Single model or ensemble configuration (similar to inference_config or ensemble_inference_config)
model: # if single model
  type: "vit"
  checkpoint: "/path/to/model.pth"
  params: { ... }
# ensemble: # if ensemble
#   method: "average"
#   models: [ ... ]

video:
  sample_rate: 10 # Process one frame every 10 frames
  batch_size: 16
  # ... other video settings
```

Run:

```bash
python inference/video_inference.py \
  --config configs/video_inference_config.yaml \
  --video /path/to/input_video.mp4 \
  --output /path/to/annotated_video.mp4 # Optional: save annotated video
  --analysis /path/to/analysis_report_video.mp4 # Optional: save analysis viz
```

*(The `video_inference.py` script defines `VideoDetector`)*

### API Deployment

The project includes a FastAPI application for serving deepfake detection.

1.  Configure `inference/deployment/config/deployment_config.yaml` with server settings, model paths (single or ensemble), etc.
2.  Run the server:
    ```bash
    python inference/deployment/server.py --config inference/deployment/config/deployment_config.yaml
    ```
    *(This uses `inference/deployment/api.py` to create the API and `server.py` to run it)*
3.  The API will be available, typically at `http://localhost:8000`. Endpoints include:
      * `POST /detect/image`: Upload an image for detection.
      * `POST /detect/video`: Upload a video for detection (background processing).
      * `GET /detect/video/status/{task_id}`: Check video processing status.
      * `GET /detect/video/result/{video_id}`: Get analysis video.
        *(Refer to `api.py` for more details on request/response formats)*

## In-Depth Evaluation (API Example from original README)

The `evaluation/analyze_results.py` module provides a `ModelEvaluator` class that can be used programmatically:

```python
from evaluation.analyze_results import ModelEvaluator #
from torch.utils.data import DataLoader
import torch # Added for device and model placeholder

# Assume 'model' is your trained PyTorch model loaded on the correct device
# model = YourTrainedModel()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device).eval()

# Assume 'test_loader' is your PyTorch DataLoader for the test set
# test_loader = DataLoader(your_test_dataset, batch_size=64)

# evaluator = ModelEvaluator(model, device=device) #
# metrics = evaluator.evaluate(test_loader) #
# print(metrics)

# report = evaluator.generate_classification_report() #
# print(report)

# figures = evaluator.visualize_results(output_dir="./evaluation_figures") #
# if figures.get("roc_curve"):
#   figures["roc_curve"].savefig("roc_curve.png")
# if figures.get("confusion_matrix"):
#   figures["confusion_matrix"].savefig("confusion_matrix.png")
```

## Extensibility

The project is designed to be extensible:

  * **Add Models**:
    1.  Implement your model class, inheriting from `models.base_model.BaseModel` if desired.
    2.  Place the model definition in an appropriate subdirectory within `models/`.
    3.  Register your model in `models.model_zoo.model_factory.py` by adding it to the `create_model` function.
    4.  Add a configuration template for your model in `configs/model_config.yaml` and potentially a model-specific config in `models/<your_model>/config/`.
  * **Custom Datasets**:
    1.  Create a new `Dataset` class in `data/datasets/`, similar to `FaceForensicsDataset` or `CelebDFDataset`.
    2.  Update `data.dataloaders.dataloader_factory.py` to support your new dataset.
    3.  Add relevant path configurations to `configs/data_config.yaml`.
  * **New Metrics**:
    1.  Implement the metric calculation in a new file within `evaluation/metrics/`.
    2.  Import and use it in `evaluation/analyze_results.py` or `evaluation/metrics/classification_metrics.py`.
  * **Fusion Techniques**:
    1.  For ensemble methods, add a new class in `fusion/ensemble/` (e.g., similar to `SimpleVoting`).
    2.  For feature fusion, add a new class in `fusion/feature_fusion/` (e.g., similar to `ConcatFusion`).
    3.  Update `configs/fusion_config.yaml` to include your new method as an option.
  * **Augmentations**:
    1.  Add new augmentation functions in `data/augmentation/`.
    2.  Integrate them into the transform compositions in `data/augmentation/color_augmentations.py`, `quality_degradation.py`, or `spatial_augmentations.py`, or by modifying how transforms are chosen in dataset classes or `data_config.yaml`.

## Notebooks

The `notebooks/` directory contains Jupyter notebooks for various tasks:

  * **`data_exploration.ipynb`**: Tools for exploring and visualizing the datasets (FaceForensics++, Celeb-DF). Includes analysis of class distributions, sample image visualization, image quality metrics (brightness, contrast, blur), and basic face landmark analysis.
  * **`demo.ipynb`**: An interactive demonstration of the deepfake detection system. It allows loading pretrained models, analyzing single images and videos frame-by-frame, and visualizing model decisions and explanations.
  * **`model_visualization.ipynb`**: Focuses on interpreting model behavior through visualizations such as model architecture diagrams, attention maps for transformer models (ViT, DeiT), Grad-CAM analysis, and t-SNE/PCA for feature space visualization.
  * **`results_analysis.ipynb`**: Provides comprehensive tools for analyzing and comparing evaluation results from multiple models and datasets. Includes visualization of metrics, ROC/PR curves, error analysis, cross-dataset generalization, confidence analysis, and model complementarity assessment.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Author

**Swarnajit Roy © 2025**
Okay, I can help you create a more detailed README.md for your Deepfake Detection project.

I will use the information from the `README.md` you provided, the YAML configuration files (e.g., `data_config.yaml`, `evaluation_config.yaml`, `fusion_config.yaml`, `model_config.yaml`, `training_config.yaml`, `deployment_config.yaml`), and the `directory_structure.txt` file to build a comprehensive overview.

Here is the enhanced README.md:

```markdown
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
├── directory\_structure.txt
├── configs/                \# Main configuration files (YAML) for various stages
│   ├── data\_config.yaml
│   ├── evaluation\_config.yaml
│   ├── fusion\_config.yaml
│   ├── model\_config.yaml
│   └── training\_config.yaml
├── data/                   \# Data loading, preprocessing, and augmentation modules
│   ├── augmentation/       \# Image augmentation techniques (color, quality, spatial)
│   ├── config/             \# Dataset-specific configurations
│   ├── dataloaders/        \# Dataloader factory
│   ├── datasets/           \# Dataset classes (FaceForensics++, Celeb-DF, custom)
│   └── preprocessing/      \# Face/frame extraction, normalization
├── evaluation/             \# Model evaluation and visualization tools
│   ├── analyze\_results.py  \# Core evaluation logic and metrics calculation
│   ├── config/             \# Evaluation-specific configurations
│   ├── cross\_dataset/      \# Cross-dataset evaluation scripts
│   ├── metrics/            \# Implementation of various evaluation metrics
│   └── visualisation/      \# Visualization scripts (ROC, Grad-CAM, attention, etc.)
├── fusion/                 \# Ensemble and feature fusion methods
│   ├── config/             \# Fusion-specific configurations
│   ├── ensemble/           \# Ensemble methods (voting, stacking, weighted averaging)
│   └── feature\_fusion/     \# Feature fusion techniques (attention, concatenation, transformer)
├── inference/              \# Inference scripts and deployment tools
│   ├── config/             \# Inference and deployment configurations
│   ├── deployment/         \# API deployment using FastAPI
│   │   ├── api.py
│   │   ├── server.py
│   │   └── ...
│   ├── ensemble\_inference.py \# Script for ensemble-based inference
│   ├── inference.py        \# Script for single model inference
│   └── video\_inference.py  \# Script for video-based inference
├── models/                 \# Model definitions
│   ├── base\_model.py       \# Base model class with common utilities
│   ├── deit/               \# DeiT model implementation
│   ├── model\_zoo/          \# Model factory for creating models
│   ├── swin/               \# Swin Transformer model implementation
│   └── vit/                \# Vision Transformer (ViT) model implementation
├── notebooks/              \# Jupyter notebooks for exploration, demo, and analysis
│   ├── data\_exploration.ipynb    \# Notebook for exploring datasets
│   ├── demo.ipynb                \# Interactive demo of the detection system
│   ├── model\_visualization.ipynb \# Notebook for visualizing model internals
│   └── results\_analysis.ipynb    \# Notebook for analyzing evaluation results
├── scripts/                \# Helper scripts for various tasks
│   ├── download\_faceforensics.py \# Script to download FaceForensics++ dataset
│   ├── evaluate\_all.py         \# Script to evaluate all trained models
│   ├── preprocess\_celebdf.py     \# Script to preprocess Celeb-DF dataset
│   ├── preprocess\_faceforensics.py \# Script to preprocess FaceForensics++ dataset
│   ├── train\_ensemble.py       \# Script to train ensemble models
│   ├── train\_individual.py     \# Script to train individual models
│   └── visualize\_results.py    \# Script to generate plots from evaluation results
├── training/               \# Training components
│   ├── callbacks/          \# Training callbacks (EarlyStopping, ModelCheckpoint)
│   ├── config/             \# Training-specific configurations
│   ├── losses/             \# Custom loss functions (Focal, Contrastive, Triplet)
│   ├── optimizers/         \# Optimizer factory
│   ├── schedulers/         \# Learning rate scheduler factory
│   ├── train.py            \# Main training script using Trainer class
│   └── trainer.py          \# Core Trainer class for managing training loop
└── utils/                  \# Utility functions
├── config\_utils.py     \# Utilities for loading/saving YAML configs
├── distributed\_utils.py \# Utilities for distributed training
├── file\_utils.py       \# File system utilities
├── logging\_utils.py    \# Logging setup and AverageMeter
└── visualization\_utils.py \# Basic visualization utilities

````

## Configuration Files

The project heavily relies on YAML configuration files located in the `configs/` directory and sub-directories like `data/config/`, `evaluation/config/`, etc. These files control various aspects of the project:

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
        This script processes original and manipulated sequences (Deepfakes, Face2Face, FaceSwap, NeuralTextures).
    * **Celeb-DF**:
        ```bash
        python scripts/preprocess_celebdf.py --input /path/to/CelebDF --output /path/to/processed/CelebDF
        ```
        This script processes real and synthetic videos from Celeb-DF.

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
````

**Example using a configuration file (recommended):**
First, create or modify a configuration file, for example, `configs/training_config.yaml`. Ensure paths and parameters are set correctly.

```yaml
# configs/training_config.yaml (example snippet)
experiment:
  name: "deepfake_detection_vit_celebdf"
  output_dir: "experiments/vit_celebdf_run1"
  seed: 42
data:
  img_size: 224
  celebdf_root: "/path/to/processed/CelebDF" # Ensure this points to preprocessed data
  dataset: "celebdf"
  # ... other data settings
model:
  type: "vit" # e.g., vit, deit, swin
  # ... model-specific params from configs/model_config.yaml
training:
  batch_size: 32
  epochs: 50
  optimizer:
    type: "adam"
    lr: 0.0001
  # ... other training settings
logging:
  tensorboard: true
  save_dir: "logs" # Relative to experiment output_dir
  # ...
```

Then run the training using `training/train.py`:

```bash
python training/train.py \
  --config configs/training_config.yaml \
  --output ./trained_models/vit_celebdf_experiment # Overrides output_dir in config if needed
```

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

Alternatively, refer to `configs/fusion_config.yaml` for ensemble configurations (e.g., `method: "stacking"` or `"weighted_average"`) and adapt `scripts/train_ensemble.py` or a dedicated fusion training script.

#### c. Training Feature Fusion Models

Feature fusion models combine features from multiple base models at an intermediate level.
Configure this using `configs/fusion_config.yaml` by setting `mode: "feature_fusion"` and specifying the fusion method (e.g., `attention`, `concat`, `transformer`) and base models.

```yaml
# configs/fusion_config.yaml (example snippet for feature fusion)
experiment:
  name: "deepfake_detection_feature_fusion"
  output_dir: "fusion_results/attention_vit_deit"
mode: "feature_fusion"
dataset: "celebdf"
data:
  celebdf_root: "/path/to/processed/CelebDF"
  # ...
fusion:
  method: "attention" # Options: concat, attention, transformer, bottleneck
  fusion_dim: 512
  models:
    - type: "vit"
      checkpoint: "/path/to/checkpoints/vit_best.pth"
      feature_dim: 768
      feature_layer: "cls_token" # Or other appropriate layer
    - type: "deit"
      checkpoint: "/path/to/checkpoints/deit_best.pth"
      feature_dim: 768
      feature_layer: "cls_token"
  architecture:
    attention: # Parameters specific to attention fusion
      num_heads: 8
      dropout: 0.1
# ... training settings for the fusion model
```

A dedicated training script for fusion models, similar to `training/train.py` but using `fusion_config.yaml`, would be used.

### 2\. Evaluating Models

Use `scripts/evaluate_all.py` to evaluate trained models on specified datasets. This script calculates various metrics and can generate visualizations.

**Example:**

```bash
python scripts/evaluate_all.py \
  --data_dir /path/to/processed/data \
  --model_paths ./trained_models/model1.pth ./trained_models/another_model.pth \
  --datasets faceforensics celebdf \
  --output_dir ./evaluation_results
```

This script uses `evaluation/analyze_results.py` internally. Configuration for evaluation can also be managed via `configs/evaluation_config.yaml`, which specifies models, datasets, metrics, and visualization options.

### 3\. Visualizing Evaluation Results

Use `scripts/visualize_results.py` to generate plots and visualizations from the output of `evaluate_all.py`.

**Example:**

```bash
python scripts/visualize_results.py \
  --results_dir ./evaluation_results \
  --output_dir ./visualization_output
```

This script will typically generate ROC curves, precision-recall curves, confusion matrices, and comparative metric plots.

### 4\. Running Inference

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

### 5\. API Deployment

The project includes a FastAPI application for deploying the deepfake detection system as a REST API.

1.  **Configure Deployment**: Modify `inference/deployment/config/deployment_config.yaml` to set server options, model paths (single or ensemble), and other API behaviors.

2.  **Run Server**:

    ```bash
    python inference/deployment/server.py --config inference/deployment/config/deployment_config.yaml
    ```

    The API server will start (default: `http://0.0.0.0:8000`). Endpoints will be available for image and video detection, including options for explanations. See `inference/deployment/api.py` for endpoint details.

    Key endpoints typically include:

      * `POST /detect/image`: Upload an image for detection.
      * `POST /detect/video`: Upload a video for detection (runs as a background task).
      * `GET /detect/video/status/{task_id}`: Check video processing status.
      * `GET /detect/video/result/{video_id}`: Retrieve annotated video.

## In-Depth Evaluation (API Example from original README)

The `evaluation/analyze_results.py` module provides a `ModelEvaluator` class for programmatic evaluation.

```python
from evaluation.analyze_results import ModelEvaluator
from torch.utils.data import DataLoader
import torch # Assuming torch is imported

# Assume:
# model = Your trained PyTorch model (loaded and set to device)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# test_loader = DataLoader for your test dataset

# Example (ensure model and test_loader are defined)
# model = create_model(model_type='vit', **vit_params) # Example
# model.load_state_dict(torch.load('path/to/your/model.pth')['model'])
# model.to(device).eval()
# test_dataset = ... # Your test dataset instance
# test_loader = DataLoader(test_dataset, batch_size=64)


evaluator = ModelEvaluator(model, device=device)
metrics = evaluator.evaluate(test_loader) # Pass your DataLoader
print("Calculated Metrics:", metrics)

report = evaluator.generate_classification_report() # Generates sklearn classification report
print("Classification Report:\n", report)

# Visualize results (e.g., ROC curve, confusion matrix)
# Ensure output_dir exists or is handled
figures = evaluator.visualize_results(output_dir="./evaluation_figures")
if "roc_curve" in figures and figures["roc_curve"] is not None:
    figures["roc_curve"].savefig("roc_curve.png")
if "confusion_matrix" in figures and figures["confusion_matrix"] is not None:
    figures["confusion_matrix"].savefig("confusion_matrix.png")
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

## Author

**Swarnajit Roy © 2025**

```
```
