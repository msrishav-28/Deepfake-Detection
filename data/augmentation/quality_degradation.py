# data/augmentation/quality_degradation.py
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_quality_transforms(img_size=224):
    """Get quality degradation transforms"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.OneOf([
            A.ImageCompression(
                quality_lower=50,
                quality_upper=90,
                p=1.0
            ),
            A.GaussianBlur(
                blur_limit=(3, 7),
                p=1.0
            ),
            A.GaussNoise(
                var_limit=(10.0, 50.0),
                p=1.0
            ),
            A.MotionBlur(
                blur_limit=(7, 15),
                p=1.0
            )
        ], p=0.8),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])