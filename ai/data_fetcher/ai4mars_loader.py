"""
AI4Mars Dataset Loader
Downloads and processes the AI4Mars dataset for terrain classification
Dataset (NASA data portal): https://data.nasa.gov/d/cykx-2qix
"""

import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Optional, List, Tuple
import logging
import shutil
from tqdm import tqdm
logger = logging.getLogger(__name__)


class AI4MarsLoader:
    """
    Loader for AI4Mars dataset
    """

    DATA_SOURCES = {
        'full': {
            'url': 'https://data.nasa.gov/download/cykx-2qix/application%2Fzip',
            'filename': 'ai4mars-dataset-merged-0.1.zip',
            'size': '5.72 GB',
            'description': 'Complete AI4Mars dataset (merged annotations)',
            'is_archive': True
        },
        'sample': {
            'url': 'https://data.nasa.gov/api/views/cykx-2qix/rows.csv?accessType=DOWNLOAD',
            'filename': 'ai4mars-sample.csv',
            'size': '120 MB',
            'description': 'Sample CSV export for lightweight experiments',
            'is_archive': False
        }
    }
    
    # Terrain classes
    CLASSES = {
        0: 'soil',
        1: 'bedrock',
        2: 'sand',
        3: 'big_rock',
    }
    
    CLASS_COLORS = {
        0: (139, 69, 19),    # soil - brown
        1: (128, 128, 128),  # bedrock - gray
        2: (255, 228, 181),  # sand - tan
        3: (105, 105, 105),  # big_rock - dark gray
    }
    
    def __init__(
        self,
        root_dir: str = "data/raw/mars_dataset",
        version: str = 'full'
    ):
        """
        Initialize loader

        Args:
            root_dir: Root directory for dataset
            version: 'full' or 'sample'
        """
        # Normalize root dir and store
        self.root_dir = Path(root_dir)
        self.version = version

        if version not in self.DATA_SOURCES:
            raise ValueError(f"Invalid version: {version}. Choose 'full' or 'sample'")

        self.root_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"AI4MarsLoader initialized: {self.root_dir}")
        logger.info(f"Version: {version} - {self.DATA_SOURCES[version]['description']}")

    def download(self, force: bool = False) -> Path:
        """
        Download dataset
        
        Args:
            force: Force redownload even if exists
            
        Returns:
            Path to downloaded archive
        """
        source = self.DATA_SOURCES[self.version]
        url = source['url']
        filename = source['filename']
        filepath = self.root_dir / filename

        # Check if already downloaded
        if filepath.exists() and not force:
            logger.info(f"Dataset already downloaded: {filepath}")
            return filepath

        # Download
        logger.info(f"Downloading AI4Mars dataset ({source['size']})...")
        logger.info(f"URL: {url}")

        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                with tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    desc=filename
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info(f"Download complete: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Download failed: {e}")
            if filepath.exists():
                filepath.unlink()
            raise

    def extract(self, archive_path: Optional[Path] = None, force: bool = False) -> Path:
        """
        Extract dataset archive

        Args:
            archive_path: Path to archive (None = use default)
            force: Force re-extraction
            
        Returns:
            Path to extracted directory
        """
        if archive_path is None:
            filename = self.DATA_SOURCES[self.version]['filename']
            archive_path = self.root_dir / filename

        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")

        source = self.DATA_SOURCES[self.version]

        # Non-archive assets (e.g., CSV exports) do not require extraction
        if not source['is_archive']:
            logger.info("No extraction required for non-archive dataset")
            return archive_path

        # Determine extraction directory
        extract_dir = self.root_dir / 'extracted'

        # Check if already extracted
        if extract_dir.exists() and not force:
            logger.info(f"Dataset already extracted: {extract_dir}")
            return extract_dir
        
        logger.info(f"Extracting {archive_path}...")
        
        try:
            # Create extraction directory
            extract_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract based on file type
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                    
            elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_dir)
                    
            else:
                raise ValueError(f"Unsupported archive format: {archive_path.suffix}")
            
            logger.info(f"Extraction complete: {extract_dir}")
            return extract_dir
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            raise

    def get_dataset_path(self) -> Path:
        """Get path to extracted dataset"""
        source = self.DATA_SOURCES[self.version]

        if not source['is_archive']:
            raise FileNotFoundError(
                "Sample dataset is provided as a CSV export and does not contain imagery. "
                "Use the 'full' dataset or convert the CSV into image/label pairs before loading."
            )

        extract_dir = self.root_dir / 'extracted'
        
        # Find the actual dataset directory (might be nested)
        if not extract_dir.exists():
            raise FileNotFoundError(
                f"Extraction directory not found: {extract_dir}. Run download() and extract() first."
            )

        candidate_patterns = [
            'msl-labeled-data*',
            'ai4mars-dataset*',
            '*merged*',
            '*ai4mars*'
        ]

        dataset_dirs: List[Path] = []
        for pattern in candidate_patterns:
            dataset_dirs.extend(
                [path for path in extract_dir.glob(pattern) if path.is_dir()]
            )

        # As a fallback, look one level deeper for directories containing splits
        if not dataset_dirs:
            for subdir in extract_dir.iterdir():
                if subdir.is_dir():
                    for pattern in candidate_patterns:
                        dataset_dirs.extend(
                            [path for path in subdir.glob(pattern) if path.is_dir()]
                        )
                    if (subdir / 'train').exists():
                        dataset_dirs.append(subdir)

        if not dataset_dirs:
            raise FileNotFoundError(
                f"Dataset not found in {extract_dir}. Ensure the NASA archive structure is preserved."
            )
        
        return dataset_dirs[0]

    def get_image_list(self, split: Optional[str] = None) -> List[Path]:
        """
        Get list of image files
        
        Args:
            split: 'train', 'val', or None (all)
            
        Returns:
            List of image paths
        """
        dataset_path = self.get_dataset_path()
        
        if split:
            split_dir = dataset_path / split
            if not split_dir.exists():
                raise ValueError(f"Split not found: {split}")
            image_paths = list(split_dir.glob('**/*.jpg'))
        else:
            image_paths = list(dataset_path.glob('**/*.jpg'))
        
        logger.info(f"Found {len(image_paths)} images (split={split})")
        return sorted(image_paths)

    def get_label_path(self, image_path: Path) -> Path:
        """Get corresponding label path for an image"""
        # Labels are typically in same directory with different extension
        label_path = image_path.with_suffix('.png')
        
        if not label_path.exists():
            # Try alternate naming
            label_path = image_path.parent / f"{image_path.stem}_label.png"
        
        return label_path

    def get_statistics(self) -> dict:
        """Get dataset statistics"""
        try:
            dataset_path = self.get_dataset_path()
            
            images = self.get_image_list()
            
            stats = {
                'total_images': len(images),
                'dataset_path': str(dataset_path),
                'version': self.version,
                'classes': self.CLASSES,
                'splits': {}
            }
            
            # Check for splits
            for split in ['train', 'val', 'test']:
                split_images = self.get_image_list(split) if (dataset_path / split).exists() else []
                stats['splits'][split] = len(split_images)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}

    def create_split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42
    ):
        """
        Create train/val/test splits
        
        Args:
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            random_seed: Random seed for reproducibility
        """
        import random
        
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        
        dataset_path = self.get_dataset_path()
        all_images = list(dataset_path.glob('**/*.jpg'))
        
        # Shuffle
        random.seed(random_seed)
        random.shuffle(all_images)
        
        # Calculate split points
        n_total = len(all_images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_images = all_images[:n_train]
        val_images = all_images[n_train:n_train + n_val]
        test_images = all_images[n_train + n_val:]
        
        # Create split directories
        for split, images in [
            ('train', train_images),
            ('val', val_images),
            ('test', test_images)
        ]:
            split_dir = dataset_path / split
            split_dir.mkdir(exist_ok=True)
            
            # Move images
            for img in images:
                dest = split_dir / img.name
                if not dest.exists():
                    shutil.copy2(img, dest)
                
                # Also copy label if exists
                label = self.get_label_path(img)
                if label.exists():
                    label_dest = split_dir / label.name
                    if not label_dest.exists():
                        shutil.copy2(label, label_dest)
        
        logger.info(f"Created splits: train={len(train_images)}, val={len(val_images)}, test={len(test_images)}")


# Convenience fun
