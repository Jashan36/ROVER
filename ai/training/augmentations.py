"""
Data Augmentation for Mars Terrain Images
Realistic augmentations that preserve terrain characteristics
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from typing import Callable
import logging

logger = logging.getLogger(__name__)


class MarsAugmentation:
    """
    Mars-specific augmentation pipeline
    """
    
    @staticmethod
    def get_training_transforms(
        image_size: tuple = (512, 512),
        p_geometric: float = 0.5,
        p_color: float = 0.5
    ) -> A.Compose:
        """
        Get training augmentations
        
        Args:
            image_size: Target image size
            p_geometric: Probability of geometric transforms
            p_color: Probability of color transforms
            
        Returns:
            Albumentations composition
        """
        transforms = A.Compose([
            # Geometric transforms
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=45,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=255,
                p=p_geometric
            ),
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=255,
                p=0.2
            ),
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.3,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=255,
                p=0.2
            ),
            
            # Color transforms (maintain Mars-like appearance)
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=p_color
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,  # Small hue shift to maintain Mars colors
                sat_shift_limit=20,
                val_shift_limit=20,
                p=p_color
            ),
            A.RGBShift(
                r_shift_limit=15,
                g_shift_limit=15,
                b_shift_limit=15,
                p=p_color
            ),
            A.ChannelShuffle(p=0.1),
            
            # Noise and blur (simulate sensor effects)
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
            ], p=0.3),
            
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.2),
            
            # Shadow and lighting (Mars has strong shadows)
            A.RandomShadow(
                shadow_roi=(0, 0, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=5,
                p=0.3
            ),
            
            # Coarse dropout (simulate rocks/shadows)
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                mask_fill_value=255,
                p=0.3
            ),
            
            # Ensure correct size
            A.Resize(image_size[0], image_size[1], interpolation=cv2.INTER_LINEAR),
        ], additional_targets={'mask': 'mask'})
        
        return transforms

    @staticmethod
    def get_validation_transforms(
        image_size: tuple = (512, 512)
    ) -> A.Compose:
        """
        Get validation augmentations (minimal/none)
        
        Args:
            image_size: Target image size
            
        Returns:
            Albumentations composition
        """
        transforms = A.Compose([
            A.Resize(image_size[0], image_size[1], interpolation=cv2.INTER_LINEAR),
        ], additional_targets={'mask': 'mask'})
        
        return transforms

    @staticmethod
    def get_test_time_augmentation() -> A.Compose:
        """
        Get test-time augmentation (TTA)
        Apply multiple augmentations and average predictions
        """
        # TTA with horizontal flip
        transforms = A.Compose([
            A.HorizontalFlip(p=1.0),
        ])
        
        return transforms


# Convenience functions
def get_training_augmentations(
    image_size: tuple = (512, 512),
    intensity: str = 'medium'
) -> Callable:
    """
    Get training augmentations with preset intensity
    
    Args:
        image_size: Target size
        intensity: 'light', 'medium', or 'heavy'
        
    Returns:
        Augmentation function
    """
    intensity_map = {
        'light': (0.3, 0.3),
        'medium': (0.5, 0.5),
        'heavy': (0.7, 0.7)
    }
    
    p_geometric, p_color = intensity_map.get(intensity, (0.5, 0.5))
    
    return MarsAugmentation.get_training_transforms(
        image_size=image_size,
        p_geometric=p_geometric,
        p_color=p_color
    )


def get_validation_augmentations(
    image_size: tuple = (512, 512)
) -> Callable:
    """Get validation augmentations"""
    return MarsAugmentation.get_validation_transforms(image_size=image_size)


def get_classification_training_augmentations(
    image_size: tuple = (512, 512),
    intensity: str = "medium",
) -> Callable:
    """Augmentations for classification-only datasets (image transforms only)."""
    intensity_map = {
        "light": (0.3, 0.3),
        "medium": (0.5, 0.5),
        "heavy": (0.7, 0.7),
    }
    p_geometric, p_color = intensity_map.get(intensity, (0.5, 0.5))

    transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale=(0.9, 1.1),
            rotate=(-30, 30),
            shear=(-10, 10),
            p=p_geometric,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=p_color,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=p_color,
        ),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.3),
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
            A.GaussianBlur(blur_limit=3, p=1.0),
        ], p=0.2),
        A.CoarseDropout(
            max_holes=6,
            max_height=32,
            max_width=32,
            min_holes=1,
            min_height=8,
            min_width=8,
            fill_value=0,
            p=0.3,
        ),
        A.Resize(image_size[0], image_size[1], interpolation=cv2.INTER_LINEAR),
    ])
    return transforms


def get_classification_validation_augmentations(
    image_size: tuple = (512, 512)
) -> Callable:
    """Validation pipeline for classification."""
    return A.Compose([
        A.Resize(image_size[0], image_size[1], interpolation=cv2.INTER_LINEAR),
    ])


# Testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Augmentation Test")
    print("=" * 60)
    
    # Create dummy image and mask
    image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    mask = np.random.randint(0, 5, (512, 512), dtype=np.uint8)
    
    # Get augmentations
    train_aug = get_training_augmentations(intensity='medium')
    val_aug = get_validation_augmentations()
    
    print("\nApplying training augmentations...")
    transformed = train_aug(image=image, mask=mask)
    aug_image = transformed['image']
    aug_mask = transformed['mask']
    
    print(f"Original shape: {image.shape}")
    print(f"Augmented shape: {aug_image.shape}")
    print(f"Mask shape: {aug_mask.shape}")
    print(f"Mask unique values: {np.unique(aug_mask)}")
    
    print("\nApplying validation augmentations...")
    transformed_val = val_aug(image=image, mask=mask)
    
    print("Augmentations loaded successfully!")
