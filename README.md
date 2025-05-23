# Deepfake Detection Project — README

## Overview

This project implements deepfake detection using various deep learning models and fusion techniques. It includes tools for:

- Preprocessing datasets (FaceForensics++, Celeb-DF)
- Training individual and ensemble models
- Evaluating and visualizing model performance

## Directory Structure

```
x/
├── LICENSE
├── README.md
├── scripts/                # Preprocessing, training, evaluation, visualization
├── evaluation/            # Model evaluation and visualization
├── models/                # Model definitions
├── training/              # Training components
├── fusion/                # Ensemble and feature fusion methods
├── data/                  # Data loading, preprocessing, augmentation
└── utils/                 # Utility functions
```

## Key Modules

### `scripts/`
- `train_individual.py`: Train individual models (EfficientNet, ResNet, Xception)
- `train_ensemble.py`: Train ensemble models
- `evaluate_all.py`: Evaluate models
- `visualize_results.py`: Generate evaluation plots
- `preprocess_celebdf.py`, `preprocess_faceforensics.py`: Dataset preprocessing

### `evaluation/`
- `analyze_results.py`: Evaluation logic
- `visualization/`: ROC curves, Grad-CAM, PCA/TSNE visualizations
- `metrics/`: Accuracy, F1, ROC, confusion matrix calculations
- `cross_dataset/`: Cross-dataset evaluation support
- `config/`: Evaluation configurations

### `models/`
- Vision Transformers: ViT, DeiT, Swin
- CNNs (placeholders): EfficientNet, ResNet, Xception
- `ensemble.py`: Ensemble model definition
- `model_zoo/`: Model factory interface

### `training/`
- `trainer.py`: Training loop
- `callbacks/`, `losses/`, `optimizers/`, `schedulers/`: Modular components
- `config/`: Training YAML configurations

### `fusion/`
- `ensemble/`: Voting, stacking, weighted averaging
- `feature_fusion/`: Attention, concatenation, transformer fusion
- `config/`: Fusion configuration YAML

### `data/`
- `datasets/`: FaceForensics++, Celeb-DF, and custom datasets
- `preprocessing/`: Face and frame extraction, normalization
- `augmentation/`: Spatial, color, quality augmentations
- `dataloaders/`: Dataloader factory
- `config/`: Dataset configuration

### `utils/`
- Logging, checkpointing, metric utilities, config loaders

---

## Setup

1. **Clone Repository**
   ```bash
   git clone <repository_url>
   cd x
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download and Preprocess Datasets**

   - **FaceForensics++**
     [Official Site](https://faceforensics.org/)
     ```bash
     python scripts/preprocess_faceforensics.py --input /path/to/FaceForensics --output /path/to/processed/FaceForensics
     ```

   - **Celeb-DF**
     [GitHub Repo](https://github.com/danmohaha/celeb-deepfakeforensics)
     ```bash
     python scripts/preprocess_celebdf.py --input /path/to/CelebDF --output /path/to/processed/CelebDF
     ```

---

## Usage

### Train Individual Models
```bash
python scripts/train_individual.py   --data_dir /path/to/processed/data   --model_type efficientnet   --dataset faceforensics   --output_dir ./trained_models
```

### Train Ensemble Models
```bash
python scripts/train_ensemble.py   --data_dir /path/to/processed/data   --model_paths ./model1.pth ./model2.pth   --dataset combined   --output_dir ./trained_models   --freeze_base_models
```

### Evaluate Models
```bash
python scripts/evaluate_all.py   --data_dir /path/to/processed/data   --model_paths ./model1.pth   --datasets faceforensics celebdf   --output_dir ./evaluation_results
```

### Visualize Evaluation Results
```bash
python scripts/visualize_results.py   --results_dir ./evaluation_results   --output_dir ./visualization_results
```

---

## Using Configuration Files

Scripts can accept YAML config files from their respective `config/` folders.

Example (training):
```bash
python training/train.py   --config training/config/training_config.yaml   --output ./trained_model
```

---

## In-Depth Evaluation (API Example)

```python
from evaluation.analyze_results import ModelEvaluator
from torch.utils.data import DataLoader

model = ...  # Your trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

test_loader = DataLoader(..., batch_size=64)

evaluator = ModelEvaluator(model, device=device)
metrics = evaluator.evaluate(test_loader)
print(metrics)

report = evaluator.generate_classification_report()
print(report)

figures = evaluator.visualize_results(output_dir="./evaluation_figures")
figures["roc_curve"].savefig("roc_curve.png")
figures["confusion_matrix"].savefig("confusion_matrix.png")
```

---

## Extensibility

- **Add Models**: Implement in `models/`, register via `model_factory.py`
- **Custom Datasets**: Define in `data/datasets/`
- **New Metrics**: Add in `evaluation/metrics/`, use in `analyze_results.py`
- **Fusion Techniques**: Add to `fusion/ensemble/` or `fusion/feature_fusion/`
- **Augmentations**: Extend `data/augmentation/`

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Author

**Swarnajit Roy © 2025**