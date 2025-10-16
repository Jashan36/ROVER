"""
Training Package
Tools for training terrain segmentation models
"""

from .train_segmentation import (
    train_model,
    TrainingConfig,
    Trainer
)

from .train_classification import (
    train_classification_model,
    ClassificationTrainingConfig,
    ClassificationTrainer
)

from .dataset import (
    AI4MarsDataset,
    TerrainDataset,
    MarsClassificationDataset,
    get_data_loaders
)

from .augmentations import (
    get_training_augmentations,
    get_validation_augmentations,
    get_classification_training_augmentations,
    get_classification_validation_augmentations,
    MarsAugmentation
)

__all__ = [
    # Segmentation training
    'train_model',
    'TrainingConfig',
    'Trainer',

    # Classification training
    'train_classification_model',
    'ClassificationTrainingConfig',
    'ClassificationTrainer',

    # Dataset
    'AI4MarsDataset',
    'TerrainDataset',
    'MarsClassificationDataset',
    'get_data_loaders',

    # Augmentations
    'get_training_augmentations',
    'get_validation_augmentations',
    'get_classification_training_augmentations',
    'get_classification_validation_augmentations',
    'MarsAugmentation',
]
