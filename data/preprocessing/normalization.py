# data/preprocessing/normalization.py
import cv2
import numpy as np
import torch
from torchvision import transforms

def normalize_face(face_img, target_size=224):
    """
    Normalize a face image for model input
    
    Args:
        face_img: Face image (NumPy array)
        target_size: Target size (square)
        
    Returns:
        Normalized image tensor
    """
    # Resize image
    face_img = cv2.resize(face_img, (target_size, target_size))
    
    # Convert to RGB if grayscale
    if len(face_img.shape) == 2:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
    
    # Convert from BGR to RGB if needed
    if face_img.shape[2] == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    # Convert to float
    face_img = face_img.astype(np.float32) / 255.0
    
    # Apply normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return transform(face_img)


def get_train_transforms(img_size=224):
    """Get training data transforms with augmentation"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_test_transforms(img_size=224):
    """Get test data transforms without augmentation"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])