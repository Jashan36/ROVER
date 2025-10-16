"""
Dataset Loaders for Terrain Segmentation Training
Supports AI4Mars and custom Mars imagery datasets
"""

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, Callable, List, Dict
import logging
import json
import pandas as pd
import sys
import platform

logger = logging.getLogger(__name__)


class AI4MarsDataset(Dataset):
    """
    Dataset loader for AI4Mars terrain classification dataset
    """
    
    # Class mapping
    CLASS_NAMES = ['soil', 'bedrock', 'sand', 'big_rock']
    CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    
    # Original AI4Mars uses different encoding - map to our classes
    AI4MARS_TO_CLASS = {
        0: 0,  # soil
        1: 1,  # bedrock  
        2: 2,  # sand
        3: 3,  # big_rock
        255: 4  # background/unknown
    }
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (512, 512),
        use_cache: bool = True
    ):
        """
        Initialize dataset
        
        Args:
            root_dir: Root directory containing AI4Mars data
            split: 'train', 'val', or 'test'
            transform: Augmentation transforms
            target_size: Resize images to this size
            use_cache: Cache loaded images in memory
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.target_size = target_size
        self.use_cache = use_cache
        
        # Find dataset directory
        self.dataset_dir = self._find_dataset_dir()
        
        # Get image and label pairs
        self.samples = self._load_samples()
        
        # Cache
        self._cache = {} if use_cache else None
        
        logger.info(f"AI4MarsDataset initialized: {split} split with {len(self.samples)} samples")

    def _find_dataset_dir(self) -> Path:
        """Find the extracted dataset directory"""
        # Try common locations
        possible_dirs = [
            self.root_dir / 'extracted' / 'msl-labeled-data-v1',
            self.root_dir / 'msl-labeled-data-v1',
            self.root_dir / 'extracted' / 'ai4mars-dataset-merged-0.1',
            self.root_dir / 'ai4mars-dataset-merged-0.1',
            self.root_dir / 'extracted' / self.split,
            self.root_dir / self.split
        ]

        # Add any directory beneath extracted that looks like AI4Mars content
        extracted_dir = self.root_dir / 'extracted'
        if extracted_dir.exists():
            for candidate in extracted_dir.iterdir():
                if candidate.is_dir() and 'ai4mars' in candidate.name.lower():
                    possible_dirs.append(candidate)
                    possible_dirs.append(candidate / self.split)

        for directory in possible_dirs:
            if directory.exists():
                return directory

        raise FileNotFoundError(
            f"Could not find dataset in {self.root_dir}. "
            "Please run AI4MarsLoader.download() and extract() first."
        )

    def _load_samples(self) -> List[Tuple[Path, Path]]:
        """Load list of (image, label) pairs"""
        samples = []
        
        # Find all images
        image_patterns = ['*.jpg', '*.png', '*.jpeg']
        image_paths = []
        
        for pattern in image_patterns:
            image_paths.extend(self.dataset_dir.glob(f'**/{pattern}'))
        
        # Find corresponding labels
        for img_path in image_paths:
            # Try different label naming conventions
            label_candidates = [
                img_path.with_suffix('.png'),  # Same name, .png extension
                img_path.parent / f"{img_path.stem}_label.png",
                img_path.parent / f"{img_path.stem}_mask.png",
                img_path.parent / 'labels' / f"{img_path.stem}.png"
            ]
            
            label_path = None
            for candidate in label_candidates:
                if candidate.exists():
                    label_path = candidate
                    break
            
            if label_path:
                samples.append((img_path, label_path))
            else:
                logger.debug(f"No label found for {img_path.name}")
        
        if not samples:
            logger.warning(f"No samples found in {self.dataset_dir}")
        
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get image and label pair
        
        Returns:
            Tuple of (image_tensor, label_tensor)
            image_tensor: (3, H, W) float32 in [0, 1]
            label_tensor: (H, W) int64 with class indices
        """
        # Check cache
        if self.use_cache and idx in self._cache:
            return self._cache[idx]
        
        img_path, label_path = self.samples[idx]
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load label
        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise ValueError(f"Failed to load label: {label_path}")
        
        # Resize
        if image.shape[:2] != self.target_size:
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        # Map AI4Mars labels to our class indices
        label_mapped = np.zeros_like(label)
        for ai4mars_val, class_idx in self.AI4MARS_TO_CLASS.items():
            label_mapped[label == ai4mars_val] = class_idx
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=label_mapped)
            image = transformed['image']
            label_mapped = transformed['mask']
        
        # Convert to tensors
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        if isinstance(label_mapped, np.ndarray):
            label_mapped = torch.from_numpy(label_mapped).long()
        
        result = (image, label_mapped)
        
        # Cache
        if self.use_cache:
            self._cache[idx] = result
        
        return result

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced dataset"""
        logger.info("Calculating class weights (this may take a while)...")
        
        class_counts = np.zeros(len(self.CLASS_NAMES) + 1)  # +1 for background
        
        for idx in range(len(self)):
            _, label = self[idx]
            unique, counts = torch.unique(label, return_counts=True)
            for cls, count in zip(unique, counts):
                class_counts[cls] += count.item()
        
        # Inverse frequency weights
        total_pixels = class_counts.sum()
        class_weights = total_pixels / (len(class_counts) * class_counts + 1e-6)
        
        # Normalize
        class_weights = class_weights / class_weights.sum() * len(class_counts)
        
        logger.info(f"Class weights: {class_weights}")
        
        return torch.from_numpy(class_weights).float()


class MarsClassificationDataset(Dataset):
    """
    Dataset for Mars imagery with labels provided via CSV metadata.
    Each sample returns (image_tensor, label_idx).
    """

    def __init__(
        self,
        images_dir: str,
        metadata_csv: str,
        label_column: str = "camera_name",
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (512, 512),
        normalize: bool = True,
    ):
        """
        Args:
            images_dir: Directory containing image files (e.g. train/ or val/ split).
            metadata_csv: Path to CSV with metadata including image paths and labels.
            label_column: Column name in CSV representing label.
            transform: Optional augmentation pipeline (Albumentations).
            target_size: Resize dimensions for the image.
            normalize: If True, scale images to [0, 1].
        """
        self.images_dir = Path(images_dir)
        self.metadata_csv = Path(metadata_csv)
        self.label_column = label_column
        self.transform = transform
        self.target_size = target_size
        self.normalize = normalize

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.metadata_csv.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {self.metadata_csv}")

        self._label_lookup = self._build_label_lookup()
        self.samples = self._collect_samples()
        if not self.samples:
            raise ValueError(
                f"No samples with labels found under {self.images_dir}. "
                f"Ensure the CSV includes entries for these images."
            )

        # Create class mappings
        labels = sorted({label for _, label in self.samples})
        self.class_to_idx = {label: idx for idx, label in enumerate(labels)}
        self.idx_to_class = {idx: label for label, idx in self.class_to_idx.items()}

        logger.info(
            "MarsClassificationDataset initialised: %d samples across %d classes",
            len(self.samples),
            len(self.class_to_idx),
        )

    def _build_label_lookup(self) -> Dict[str, str]:
        """Build mapping from filename to label value."""
        df = pd.read_csv(self.metadata_csv)
        if self.label_column not in df.columns:
            raise ValueError(
                f"Label column '{self.label_column}' missing from {self.metadata_csv}"
            )

        if "img_path" in df.columns:
            filenames = df["img_path"].apply(lambda p: Path(str(p)).name)
        elif "image" in df.columns:
            filenames = df["image"].apply(lambda p: Path(str(p)).name)
        else:
            filenames = df["id"].apply(lambda i: f"{i}")

        labels = df[self.label_column].astype(str)
        lookup = {}
        for filename, label in zip(filenames, labels):
            lookup[filename] = label
        return lookup

    def _collect_samples(self) -> List[Tuple[Path, str]]:
        """Gather list of (image_path, label_name) pairs for valid entries."""
        image_files = sorted(self.images_dir.glob("*.jpg"))
        image_files += sorted(self.images_dir.glob("*.png"))

        samples = []
        for img_path in image_files:
            label = self._label_lookup.get(img_path.name)
            if label is None:
                logger.debug("No label found for %s; skipping", img_path.name)
                continue
            samples.append((img_path, label))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, label_name = self.samples[idx]

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image.shape[:2] != self.target_size:
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            if self.normalize:
                image = image / 255.0

        label_idx = self.class_to_idx[label_name]
        label_tensor = torch.tensor(label_idx, dtype=torch.long)
        return image, label_tensor

    def get_class_weights(self) -> torch.Tensor:
        counts = np.zeros(len(self.class_to_idx), dtype=np.float32)
        for _, label_name in self.samples:
            counts[self.class_to_idx[label_name]] += 1

        counts[counts == 0] = 1.0
        weights = counts.sum() / counts
        weights = weights / weights.sum() * len(self.class_to_idx)
        return torch.from_numpy(weights).float()

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)


def _resolve_num_workers(requested: int) -> int:
    """Adjust num_workers to avoid multiprocessing issues on Windows/stdin."""
    if requested <= 0:
        return 0

    if platform.system() == "Windows":
        try:
            if not sys.stdin.isatty():
                logger.warning(
                    "Detected non-interactive Windows execution; forcing num_workers=0 to avoid multiprocessing issues."
                )
                return 0
        except Exception:
            logger.warning(
                "Could not determine stdin state on Windows; defaulting num_workers=0."
            )
            return 0
    return requested


class TerrainDataset(Dataset):
    """
    Generic terrain dataset for custom Mars imagery
    """
    
    def __init__(
        self,
        image_dir: str,
        label_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (512, 512)
    ):
        """
        Initialize dataset
        
        Args:
            image_dir: Directory with images
            label_dir: Directory with labels (None = unsupervised)
            transform: Augmentation transforms
            target_size: Resize to this size
        """
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir) if label_dir else None
        self.transform = transform
        self.target_size = target_size
        
        # Find images
        self.image_paths = sorted(list(self.image_dir.glob('*.jpg')) + 
                                  list(self.image_dir.glob('*.png')))
        
        logger.info(f"TerrainDataset: {len(self.image_paths)} images")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        img_path = self.image_paths[idx]
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        if image.shape[:2] != self.target_size:
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # Load label if available
        label = None
        if self.label_dir:
            label_path = self.label_dir / img_path.name
            if label_path.exists():
                label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
                if label.shape != self.target_size:
                    label = cv2.resize(label, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        # Apply transforms
        if self.transform:
            if label is not None:
                transformed = self.transform(image=image, mask=label)
                image = transformed['image']
                label = transformed['mask']
            else:
                transformed = self.transform(image=image)
                image = transformed['image']
        
        # Convert to tensors
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        if label is not None and isinstance(label, np.ndarray):
            label = torch.from_numpy(label).long()
        
        return image, label


def get_data_loaders(
    root_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    target_size: Tuple[int, int] = (512, 512),
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    dataset_type: str = "segmentation",
    metadata_csv: Optional[str] = None,
    label_column: str = "camera_name",
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create train/val/test data loaders
    
    Args:
        root_dir: Root directory with AI4Mars data
        batch_size: Batch size
        num_workers: Number of worker processes
        target_size: Image size
        train_transform: Training augmentations
        val_transform: Validation augmentations
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    dataset_type = dataset_type.lower()

    if dataset_type == "segmentation":
        train_dataset = AI4MarsDataset(
            root_dir=root_dir,
            split='train',
            transform=train_transform,
            target_size=target_size
        )
        
        val_dataset = AI4MarsDataset(
            root_dir=root_dir,
            split='val',
            transform=val_transform,
            target_size=target_size
        )
        
        test_dataset = None
        try:
            test_dataset = AI4MarsDataset(
                root_dir=root_dir,
                split='test',
                transform=val_transform,
                target_size=target_size
            )
        except Exception:
            logger.info("No test split found")

    elif dataset_type == "classification":
        root_path = Path(root_dir)
        if metadata_csv is None:
            metadata_csv = str(root_path.parent / "mars_rover_dataset.csv")

        train_dataset = MarsClassificationDataset(
            images_dir=root_path / "train",
            metadata_csv=metadata_csv,
            label_column=label_column,
            transform=train_transform,
            target_size=target_size
        )

        val_dataset = MarsClassificationDataset(
            images_dir=root_path / "val",
            metadata_csv=metadata_csv,
            label_column=label_column,
            transform=val_transform,
            target_size=target_size
        )

        test_dataset = None
        test_dir = root_path / "test"
        if test_dir.exists():
            test_dataset = MarsClassificationDataset(
                images_dir=test_dir,
                metadata_csv=metadata_csv,
                label_column=label_column,
                transform=val_transform,
                target_size=target_size
            )
        else:
            logger.info("No test split found for classification dataset")
    else:
        raise ValueError(f"Unsupported dataset_type '{dataset_type}'.")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=_resolve_num_workers(num_workers),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=_resolve_num_workers(num_workers),
        pin_memory=True
    )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=_resolve_num_workers(num_workers),
            pin_memory=True
        )
    
    return train_loader, val_loader, test_loader


# Testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Dataset Test")
    print("=" * 60)
    
    # Test dataset
    try:
        dataset = AI4MarsDataset(
            root_dir="data/raw/ai4mars",
            split='train',
            target_size=(512, 512)
        )
        
        print(f"\nDataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            image, label = dataset[0]
            print(f"Image shape: {image.shape}")
            print(f"Label shape: {label.shape}")
            print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
            print(f"Unique labels: {torch.unique(label).tolist()}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nTo use this dataset:")
        print("1. Download AI4Mars: python ai/data_fetcher/ai4mars_loader.py")
        print("2. Extract dataset")
        print("3. Run this script again")
