# data/datasets/__init__.py
from .faceforensics import FaceForensicsDataset
from .celebdf import CelebDFDataset
from .custom_dataset import DeepfakeDataset

__all__ = ["FaceForensicsDataset", "CelebDFDataset", "DeepfakeDataset"]