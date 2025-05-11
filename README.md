```markdown
# Project 'd' - Deepfake Detection

## Overview

This project focuses on developing deep learning models for detecting deepfake videos and images. It includes scripts for data preprocessing, model training (both individual models and ensembles), evaluation, and visualization of results. The project supports multiple datasets (FaceForensics++, Celeb-DF, and combined datasets) and several model architectures (EfficientNet, ResNet, Xception, ViT, DeiT, and SwinTransformer).  It also provides various evaluation metrics, visualization tools, and fusion techniques.

## Directory Structure

```
d/
├── data/                 # Data loading, preprocessing, and augmentation
│   ├── augmentation/     # Image augmentation techniques
│   ├── dataloaders/    # Custom dataloaders for various datasets
│   ├── datasets/         # Dataset definitions (FaceForensics++, Celeb-DF, etc.)
│   ├── preprocessing/    # Scripts for face extraction, frame extraction, normalization
│   └── config/         # Dataset configurations (e.g., data_config.yaml)
├── evaluation/           # Model evaluation and analysis scripts
│   ├── config/           # Evaluation configurations (e.g., eval_config.yaml)
│   ├── cross_dataset/    # Scripts for cross-dataset evaluation
│   ├── metrics/          # Scripts for calculating various evaluation metrics
│   ├── visualization/  # Visualization tools (Grad-CAM, feature visualization, etc.)
│   └── analyze_results.py  # Script to evaluate and analyze a given model 
├── fusion/               # Scripts and configurations for model fusion strategies
│   ├── config/           # Fusion configuration (e.g., fusion_config.yaml)
│   ├── ensemble/        # Scripts for ensemble methods (simple voting, weighted average, stacking)
│   └── feature_fusion/  # Scripts for feature fusion techniques (concat, attention)
├── models/               # Deep learning model definitions
│   ├── base_model.py   # Abstract class for all the models
│   ├── efficientnet.py # Implementation of efficient net
│   ├── resnet.py       # Implementation of resnet
│   ├── xception.py     # Implementation of xception
│   ├── vit/              # Vision Transformer (ViT) implementation
│   ├── deit/             # Data-efficient Image Transformer (DeiT) implementation
│   ├── swin/             # Swin Transformer implementation
│   ├── model_zoo/       # A common place to import all models
│   └── config/           # Example configuration for all models
├── scripts/              # Main scripts for training, preprocessing, and evaluation
│   ├── download_faceforensics.py
│   ├── evaluate_all.py     # Evaluate a given model over multiple datasets
│   ├── preprocess_celebdf.py
│   ├── preprocess_faceforensics.py
│   ├── train_ensemble.py  # Train an ensemble of pre-trained models
│   ├── train_individual.py # Train an individual deepfake detection model
│   └── visualize_results.py
├── training/             # Scripts for model training and related utilities
│   ├── callbacks/        # Callbacks for training (early stopping, model checkpoint)
│   ├── config/           # Training configurations (e.g., training_config.yaml)
│   ├── losses/           # Loss function definitions (Focal Loss, Contrastive Loss, etc.)
│   ├── optimizers/       # Optimizer factory
│   ├── schedulers/       # Learning rate scheduler factory
│   └── train.py          # Entry point for training a model
├── utils/                # Utility functions (logging, checkpointing, configuration loading)
└── README.md             # This file
```

## Key Scripts and Functionality

### 1. Data Preprocessing

*   **`scripts/preprocess_faceforensics.py`**: Extracts faces from FaceForensics++ videos.  Takes input and output directories as arguments, along with sampling rate and maximum number of frames.  Utilizes the `MTCNN` face detector.
*   **`scripts/preprocess_celebdf.py`**: Extracts faces from Celeb-DF videos. Similar to the FaceForensics script but tailored for the Celeb-DF dataset structure.
*   **`data/preprocessing/face_extraction.py`**: Contains functions for setting up the face detector (`setup_face_detector`) and extracting faces from video (`extract_faces_from_video`) or images (`extract_faces`).
*   **`data/preprocessing/normalization.py`**: Provides functions for normalizing face images, including resizing, color conversion, and applying standard normalization transforms (`normalize_face`, `get_train_transforms`, `get_test_transforms`).
*   **`scripts/download_faceforensics.py`:** This script is responsible for downloading parts of the FaceForensics dataset.

### 2. Model Training

*   **`scripts/train_individual.py`**: Trains individual deepfake detection models. Supports EfficientNet, ResNet, and Xception architectures.
    *   Accepts arguments for data directory, model type, output directory, batch size, epochs, learning rate, and more.
    *   Implements data transforms, dataset loading, model initialization, loss function definition, optimizer setup, and the training loop.
    *   Saves model checkpoints and training logs.
*   **`scripts/train_ensemble.py`**: Trains an ensemble of deepfake detection models.
    *   Requires paths to pre-trained individual model checkpoints.
    *   Initializes an ensemble model (`DeepfakeEnsemble`) by loading pre-trained weights.
    *   Optionally freezes the base models during training.
*   **`training/train.py`**: A modular training script that uses configurations in `.yaml` files to define model, optimizer, and data. It also supports more recent deep learning architectures such as ViT, DeiT, and SwinTransformer.
*   **`training/trainer.py`**: Contains the `Trainer` class, which encapsulates the training logic, validation, logging, and checkpointing.
*   **`training/config/training_config.yaml`**: An example configuration file defining training parameters such as batch size, learning rate, optimizer, and scheduler.

### 3. Model Evaluation

*   **`scripts/evaluate_all.py`**: Evaluates trained models on multiple datasets (FaceForensics++, Celeb-DF, and combined).
    *   Loads pre-trained models (including ensemble models).
    *   Computes various evaluation metrics (accuracy, precision, recall, F1-score, AUC, confusion matrix).
    *   Generates and saves ROC curves and Precision-Recall curves.
*   **`evaluation/analyze_results.py`**: Provides a class `ModelEvaluator` for in-depth analysis of model performance, including generating classification reports and visualizing results with ROC curves and confusion matrices.
*   **`evaluation/config/eval_config.yaml`**: Configuration file for the evaluation, including paths to datasets, model checkpoints, and evaluation metrics.
*   **`evaluation/cross_dataset/cross_evaluation.py`**: Evaluates the generalization ability of models by testing on different datasets from the one they were trained on.
*   **`evaluation/metrics/`**: Contains functions to calculate evaluation metrics.
*   **`evaluation/visualization/`**: Includes scripts for visualizing model performance, such as Grad-CAM heatmaps, feature visualization (using t-SNE or PCA), and attention maps.

### 4. Visualization

*   **`scripts/visualize_results.py`**: Creates visualizations of evaluation results.
    *   Loads evaluation results from the output directory of `evaluate_all.py`.
    *   Generates comparative bar charts for different metrics, ROC curves, confusion matrix heatmaps, and performance summary tables.
*   **`evaluation/visualization/`**: Contains modules for visualizing different aspects of model behaviour.
    *   **`grad_cam.py`**: implements Grad-CAM to visualize important regions in the image.
    *   **`feature_visualization.py`**: Provides functionalities to visualize the features learned by a model.
    *   **`attention_maps.py`**: Extracts and visualizes attention maps from Transformer-based models.

### 5. Model Fusion

*   **`fusion/`**: This directory contains various methods to combine the outputs or features of different models, such as simple voting, weighted averaging, and feature concatenation.
*   **`fusion/ensemble/`**: Contains various modules for doing ensemble-based model fusion.
*   **`fusion/feature_fusion/`**: Contains various modules for doing feature-based model fusion.

## Modules

*   **`utils/logger.py`**: Sets up logging for the training process.
*   **`utils/checkpointing.py`**: Provides functions for saving and loading model checkpoints.
*   **`utils/config_utils.py`**: Provides utility function to load configuration file.
*   **`data/datasets/`**: Contains custom dataset classes for FaceForensics++, Celeb-DF, and the combined dataset.

## Datasets

The project is designed to work with the following datasets:

*   **FaceForensics++**: A widely used benchmark dataset for deepfake detection.
*   **Celeb-DF**: A high-quality deepfake dataset with realistic manipulations.
*   **Combined**: A custom dataset that combines FaceForensics++ and Celeb-DF to increase the diversity of training data.

## Model Architectures

The project supports training and evaluation of the following model architectures:

*   **EfficientNet**: A family of efficient and accurate convolutional neural networks.
*   **ResNet**: A deep residual network architecture.
*   **Xception**: An extreme inception architecture.
*   **ViT (Vision Transformer)**: A transformer-based model for image classification.
*   **DeiT (Data-efficient Image Transformer)**: An improved version of ViT that requires less training data.
*   **Swin Transformer**: A hierarchical transformer with shifted windows for efficient and scalable vision tasks.

## Evaluation Metrics

The following evaluation metrics are used to assess the performance of the deepfake detection models:

*   Accuracy
*   Precision
*   Recall
*   F1-score
*   AUC (Area Under the ROC Curve)
*   EER (Equal Error Rate)
*   Confusion Matrix

## Usage

### 1. Data Preparation

1.  Download the FaceForensics++ and/or Celeb-DF datasets.
2.  Modify the `root` paths in the `data/config/data_config.yaml` and `training/config/training_config.yaml` files to point to the correct locations of the datasets on your system.
3.  Run the preprocessing scripts to extract faces:

```bash
python scripts/preprocess_faceforensics.py --input /path/to/FaceForensics --output /path/to/output
python scripts/preprocess_celebdf.py --input /path/to/Celeb-DF --output /path/to/output
```

### 2. Training an Individual Model

1.  Modify the `training/config/training_config.yaml` to set hyperparameters.
2.  Run the training script:

```bash
python training/train.py --config training/config/training_config.yaml --output /path/to/output
```

### 3. Training an Ensemble Model

1.  Ensure you have trained individual models and have the checkpoint paths.
2.  Modify the `scripts/train_ensemble.py` script with the correct paths to the pre-trained models.

```bash
python scripts/train_ensemble.py --data_dir /path/to/processed_data --model_paths /path/to/model1.pth /path/to/model2.pth --output_dir /path/to/ensemble_output
```

### 4. Evaluating Models

1.  Modify the `scripts/evaluate_all.py` script with the correct paths to the trained model(s).
2.  Run the evaluation script:

```bash
python scripts/evaluate_all.py --data_dir /path/to/processed_data --model_paths /path/to/model1.pth --output_dir /path/to/evaluation_output
```

### 5. Visualizing Results

1.  Run the `scripts/visualize_results.py` script, pointing it to the output directory of the evaluation script:

```bash
python scripts/visualize_results.py --results_dir /path/to/evaluation_output --output_dir /path/to/visualization_output
```

## Configuration

The project uses YAML configuration files to manage various settings:

*   **`data/config/data_config.yaml`**: Configures dataset paths, splits, and preprocessing settings.
*   **`training/config/training_config.yaml`**: Configures training parameters such as batch size, learning rate, optimizer, and scheduler.
*   **`evaluation/config/eval_config.yaml`**: Configures evaluation settings such as evaluation metrics and dataset paths.
*   **`fusion/config/fusion_config.yaml`**: Configures the parameters of various fusion methods.

## Dependencies

*   Python 3.7+
*   PyTorch 1.8+
*   Torchvision
*   NumPy
*   OpenCV
*   Tqdm
*   Scikit-learn
*   Matplotlib
*   Seaborn
*   PyYAML
*   Facenet-pytorch
*   einops
*   timm

## Installation

```bash
pip install -r requirements.txt
```

## Citation

If you use this code in your research, please cite the original authors of the datasets and model architectures used in this project.
```
