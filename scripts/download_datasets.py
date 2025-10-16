#!/usr/bin/env python3
"""
Dataset Download Script (Python)
Downloads all required datasets for Mars Rover AI system
"""

import os
import sys
import argparse
import requests
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"


class DatasetDownloader:
    """Handles dataset downloads"""
    
    def __init__(self, data_dir: Path = DATA_DIR):
        """
        Initialize downloader
        
        Args:
            data_dir: Root data directory
        """
        self.data_dir = data_dir
        self.raw_dir = data_dir / "raw"
        self.processed_dir = data_dir / "processed"
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, url: str, dest: Path, desc: str = None) -> Path:
        """
        Download file with progress bar
        
        Args:
            url: Download URL
            dest: Destination path
            desc: Progress bar description
            
        Returns:
            Path to downloaded file
        """
        desc = desc or f"Downloading {dest.name}"
        
        # Check if already exists
        if dest.exists():
            logger.info(f"{dest.name} already exists, skipping download")
            return dest
        
        logger.info(f"Downloading from {url}")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(dest, 'wb') as f, tqdm(
                desc=desc,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info(f"Downloaded to {dest}")
            return dest
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            if dest.exists():
                dest.unlink()
            raise
    
    def extract_archive(self, archive_path: Path, extract_to: Path = None) -> Path:
        """
        Extract archive
        
        Args:
            archive_path: Path to archive
            extract_to: Extraction directory (default: same as archive)
            
        Returns:
            Path to extracted directory
        """
        if extract_to is None:
            extract_to = archive_path.parent
        
        logger.info(f"Extracting {archive_path.name}...")
        
        try:
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            
            elif archive_path.suffix in ['.tar', '.gz', '.tgz', '.bz2']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_to)
            
            else:
                raise ValueError(f"Unsupported archive format: {archive_path.suffix}")
            
            logger.info(f"Extracted to {extract_to}")
            return extract_to
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            raise
    
    def download_ai4mars_sample(self) -> Path:
        """Download AI4Mars sample dataset"""
        logger.info("Downloading AI4Mars sample dataset...")
        
        url = "https://data.nasa.gov/download/cykx-2qix/application%2Fzip"
        dest_dir = self.raw_dir / "ai4mars"
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        archive = dest_dir / "ai4mars_sample.zip"
        
        try:
            # For demonstration - actual URL would be from AI4Mars
            # This is a placeholder
            logger.warning("AI4Mars sample download URL is placeholder")
            logger.info("Please manually download from: https://github.com/nasa-jpl/AI4Mars")
            
            # If we had real URL:
            # self.download_file(url, archive, "AI4Mars Sample")
            # self.extract_archive(archive, dest_dir)
            
            return dest_dir
            
        except Exception as e:
            logger.error(f"Failed to download AI4Mars: {e}")
            raise
    
    def download_perseverance_images(self, limit: int = 25) -> Path:
        """
        Download NASA Perseverance images
        
        Args:
            limit: Number of images to download
            
        Returns:
            Path to image directory
        """
        logger.info(f"Downloading {limit} Perseverance images...")
        
        dest_dir = self.raw_dir / "perseverance"
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Import NASA API module
        try:
            sys.path.insert(0, str(PROJECT_ROOT))
            from ai.data_fetcher.perseverance_api import fetch_latest_images
            
            images = fetch_latest_images(
                camera='NAVCAM_LEFT',
                limit=limit,
                download=True,
                cache_dir=str(dest_dir)
            )
            
            logger.info(f"Downloaded {len(images)} images")
            return dest_dir
            
        except ImportError as e:
            logger.error(f"Failed to import NASA API module: {e}")
            logger.info("Please ensure all dependencies are installed")
            raise
        except Exception as e:
            logger.error(f"Failed to download Perseverance images: {e}")
            raise
    
    def generate_mars_dem(self, size: tuple = (1024, 1024)) -> tuple:
        """
        Generate Mars DEM heightmap
        
        Args:
            size: Heightmap size (width, height)
            
        Returns:
            Tuple of (heightmap_path, texture_path)
        """
        logger.info("Generating Mars DEM heightmap...")
        
        dest_dir = self.raw_dir / "dem"
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            sys.path.insert(0, str(PROJECT_ROOT))
            from ai.data_fetcher.dem_processor import generate_mars_heightmap
            
            heightmap_path, texture_path = generate_mars_heightmap(
                output_dir=str(dest_dir),
                size=size,
                z_range_meters=(0.0, 10.0),
                with_texture=True
            )
            
            logger.info(f"Generated heightmap: {heightmap_path}")
            
            # Copy to simulation directory if exists
            sim_worlds = PROJECT_ROOT / "sim" / "rover_description" / "worlds"
            if sim_worlds.exists():
                import shutil
                shutil.copy(heightmap_path, sim_worlds / "mars_heightmap.png")
                if texture_path:
                    shutil.copy(texture_path, sim_worlds / "mars_texture.png")
                logger.info("Copied to simulation directory")
            
            return heightmap_path, texture_path
            
        except ImportError as e:
            logger.error(f"Failed to import DEM module: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to generate DEM: {e}")
            raise
    
    def download_pretrained_models(self) -> Path:
        """Download pretrained model weights"""
        logger.info("Checking for pretrained models...")
        
        models_dir = PROJECT_ROOT / "ai" / "models" / "weights"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = models_dir / "terrain_unet_best.pth"
        
        if model_path.exists():
            logger.info("Pretrained model already exists")
        else:
            logger.warning("No pretrained model found")
            logger.info("You will need to either:")
            logger.info("  1. Train your own: python ai/training/train_segmentation.py")
            logger.info("  2. Download pretrained weights (if available)")
        
        return models_dir
    
    def get_statistics(self) -> dict:
        """Get download statistics"""
        stats = {
            'data_dir': str(self.data_dir),
            'total_size_mb': 0,
            'datasets': {}
        }
        
        # Calculate sizes
        for item in self.raw_dir.rglob('*'):
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                stats['total_size_mb'] += size_mb
                
                # Categorize by parent directory
                dataset = item.parent.name
                if dataset not in stats['datasets']:
                    stats['datasets'][dataset] = {'files': 0, 'size_mb': 0}
                
                stats['datasets'][dataset]['files'] += 1
                stats['datasets'][dataset]['size_mb'] += size_mb
        
        return stats


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Download datasets for Mars Rover AI system'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all datasets'
    )
    parser.add_argument(
        '--minimal',
        action='store_true',
        help='Download minimal datasets (sample + DEM)'
    )
    parser.add_argument(
        '--ai4mars',
        action='store_true',
        help='Download AI4Mars sample dataset'
    )
    parser.add_argument(
        '--perseverance',
        action='store_true',
        help='Download Perseverance images'
    )
    parser.add_argument(
        '--dem',
        action='store_true',
        help='Generate Mars DEM heightmap'
    )
    parser.add_argument(
        '--models',
        action='store_true',
        help='Check pretrained models'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=25,
        help='Number of Perseverance images (default: 25)'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show download statistics'
    )
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = DatasetDownloader()
    
    print("=" * 60)
    print("MARS ROVER AI - DATASET DOWNLOADER")
    print("=" * 60)
    print()
    
    try:
        # Download based on arguments
        if args.all:
            logger.info("Downloading all datasets...")
            downloader.download_ai4mars_sample()
            downloader.download_perseverance_images(args.limit)
            downloader.generate_mars_dem()
            downloader.download_pretrained_models()
        
        elif args.minimal:
            logger.info("Downloading minimal datasets...")
            downloader.download_ai4mars_sample()
            downloader.generate_mars_dem()
            downloader.download_pretrained_models()
        
        else:
            # Individual downloads
            if args.ai4mars:
                downloader.download_ai4mars_sample()
            
            if args.perseverance:
                downloader.download_perseverance_images(args.limit)
            
            if args.dem:
                downloader.generate_mars_dem()
            
            if args.models:
                downloader.download_pretrained_models()
            
            # Show help if no arguments
            if not any([args.ai4mars, args.perseverance, args.dem, args.models]):
                parser.print_help()
                return
        
        # Show statistics
        if args.stats or args.all or args.minimal:
            print()
            print("=" * 60)
            print("DOWNLOAD STATISTICS")
            print("=" * 60)
            
            stats = downloader.get_statistics()
            print(f"\nData directory: {stats['data_dir']}")
            print(f"Total size: {stats['total_size_mb']:.2f} MB")
            print("\nDatasets:")
            for dataset, info in stats['datasets'].items():
                print(f"  {dataset}:")
                print(f"    Files: {info['files']}")
                print(f"    Size: {info['size_mb']:.2f} MB")
        
        print()
        print("✓ Dataset download completed successfully!")
        print()
        
    except KeyboardInterrupt:
        print("\n\n✗ Download cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()