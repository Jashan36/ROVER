"""
NASA Perseverance Rover API Client
Fetches real Mars images from the Mars 2020 mission
API Documentation: https://api.nasa.gov/
"""

import requests
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import time
import logging
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


@dataclass
class MarsImage:
    """Mars rover image metadata"""
    id: int
    sol: int  # Mars day
    earth_date: str
    camera: str
    img_src: str
    rover: str
    
    # Additional metadata
    sample_type: Optional[str] = None
    attitude: Optional[Dict] = None
    extended_metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    def get_filename(self) -> str:
        """Generate standardized filename"""
        # Example: sol0015_NAVCAM_LEFT_20210304_143022.jpg
        date_str = self.earth_date.replace('-', '')
        camera_clean = self.camera.replace(' ', '_').upper()
        return f"sol{self.sol:04d}_{camera_clean}_{date_str}_{self.id}.jpg"


class PerseveranceAPIClient:
    """
    Client for NASA Mars Rover Photos API
    """
    
    BASE_URL = "https://api.nasa.gov/mars-photos/api/v1/"
    
    # Camera names for Perseverance
    CAMERAS = {
        'NAVCAM_LEFT': 'Navigation Camera - Left',
        'NAVCAM_RIGHT': 'Navigation Camera - Right',
        'FRONT_HAZCAM_LEFT_A': 'Front Hazard Avoidance Camera - Left',
        'FRONT_HAZCAM_RIGHT_A': 'Front Hazard Avoidance Camera - Right',
        'REAR_HAZCAM_LEFT': 'Rear Hazard Avoidance Camera - Left',
        'REAR_HAZCAM_RIGHT': 'Rear Hazard Avoidance Camera - Right',
        'MCZ_LEFT': 'Mast Camera Zoom - Left',
        'MCZ_RIGHT': 'Mast Camera Zoom - Right',
        'SKYCAM': 'MEDA Skycam',
        'SHERLOC_WATSON': 'SHERLOC WATSON Camera',
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: str = "data/raw/perseverance"
    ):
        """
        Initialize API client
        
        Args:
            api_key: NASA API key (get free key at https://api.nasa.gov/)
                    If None, will try to read from NASA_API_KEY env variable
            cache_dir: Directory to cache downloaded images
        """
        self.api_key = api_key or os.getenv('NASA_API_KEY', 'DEMO_KEY')
        
        if self.api_key == 'DEMO_KEY':
            logger.warning(
                "Using DEMO_KEY for NASA API. "
                "Rate limited to 30 requests/hour. "
                "Get free key at https://api.nasa.gov/"
            )
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = requests.Session()
        self.session.params = {'api_key': self.api_key}
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        logger.info(f"PerseveranceAPIClient initialized. Cache: {self.cache_dir}")

    def _rate_limit(self):
        """Enforce rate limiting between requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None
    ) -> Dict:
        """
        Make API request with error handling
        
        Args:
            endpoint: API endpoint (e.g., 'rovers/perseverance/photos')
            params: Additional query parameters
            
        Returns:
            JSON response as dictionary
        """
        self._rate_limit()
        
        url = urljoin(self.BASE_URL, endpoint)
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise

    def get_photos_by_sol(
        self,
        sol: int,
        camera: Optional[str] = None,
        page: int = 1
    ) -> List[MarsImage]:
        """
        Get photos from a specific Martian sol (day)
        
        Args:
            sol: Martian sol number (0 = landing day)
            camera: Specific camera name (None = all cameras)
            page: Page number for pagination
            
        Returns:
            List of MarsImage objects
        """
        params = {
            'sol': sol,
            'page': page
        }
        
        if camera:
            params['camera'] = camera
        
        logger.info(f"Fetching photos for sol {sol}, camera={camera}, page={page}")
        
        data = self._make_request('rovers/perseverance/photos', params)
        
        photos = []
        for photo_data in data.get('photos', []):
            try:
                photo = MarsImage(
                    id=photo_data['id'],
                    sol=photo_data['sol'],
                    earth_date=photo_data['earth_date'],
                    camera=photo_data['camera']['name'],
                    img_src=photo_data['img_src'],
                    rover=photo_data['rover']['name'],
                    extended_metadata=photo_data
                )
                photos.append(photo)
            except KeyError as e:
                logger.warning(f"Failed to parse photo data: {e}")
                continue
        
        logger.info(f"Retrieved {len(photos)} photos")
        return photos

    def get_photos_by_earth_date(
        self,
        earth_date: str,
        camera: Optional[str] = None,
        page: int = 1
    ) -> List[MarsImage]:
        """
        Get photos from a specific Earth date
        
        Args:
            earth_date: Date in YYYY-MM-DD format
            camera: Specific camera name (None = all cameras)
            page: Page number
            
        Returns:
            List of MarsImage objects
        """
        params = {
            'earth_date': earth_date,
            'page': page
        }
        
        if camera:
            params['camera'] = camera
        
        logger.info(f"Fetching photos for {earth_date}, camera={camera}")
        
        data = self._make_request('rovers/perseverance/photos', params)
        
        photos = []
        for photo_data in data.get('photos', []):
            photo = MarsImage(
                id=photo_data['id'],
                sol=photo_data['sol'],
                earth_date=photo_data['earth_date'],
                camera=photo_data['camera']['name'],
                img_src=photo_data['img_src'],
                rover=photo_data['rover']['name'],
                extended_metadata=photo_data
            )
            photos.append(photo)
        
        return photos

    def get_latest_photos(
        self,
        camera: Optional[str] = None,
        limit: int = 25
    ) -> List[MarsImage]:
        """
        Get most recent photos
        
        Args:
            camera: Specific camera name
            limit: Maximum number of photos
            
        Returns:
            List of MarsImage objects
        """
        data = self._make_request('rovers/perseverance/latest_photos')
        
        photos = []
        for photo_data in data.get('latest_photos', [])[:limit]:
            if camera and photo_data['camera']['name'] != camera:
                continue
                
            photo = MarsImage(
                id=photo_data['id'],
                sol=photo_data['sol'],
                earth_date=photo_data['earth_date'],
                camera=photo_data['camera']['name'],
                img_src=photo_data['img_src'],
                rover=photo_data['rover']['name'],
                extended_metadata=photo_data
            )
            photos.append(photo)
        
        logger.info(f"Retrieved {len(photos)} latest photos")
        return photos

    def download_image(
        self,
        image: MarsImage,
        overwrite: bool = False
    ) -> Path:
        """
        Download image to cache directory
        
        Args:
            image: MarsImage object
            overwrite: Whether to overwrite existing file
            
        Returns:
            Path to downloaded image file
        """
        filename = image.get_filename()
        filepath = self.cache_dir / filename
        
        # Check if already downloaded
        if filepath.exists() and not overwrite:
            logger.debug(f"Image already cached: {filename}")
            return filepath
        
        # Download
        logger.info(f"Downloading {filename}...")
        
        try:
            self._rate_limit()
            response = requests.get(image.img_src, timeout=30, stream=True)
            response.raise_for_status()
            
            # Save image
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Save metadata
            metadata_path = filepath.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(image.to_dict(), f, indent=2)
            
            logger.info(f"Downloaded: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            raise

    def download_multiple(
        self,
        images: List[MarsImage],
        max_downloads: Optional[int] = None,
        overwrite: bool = False
    ) -> List[Path]:
        """
        Download multiple images
        
        Args:
            images: List of MarsImage objects
            max_downloads: Maximum number to download (None = all)
            overwrite: Whether to overwrite existing files
            
        Returns:
            List of paths to downloaded files
        """
        if max_downloads:
            images = images[:max_downloads]
        
        downloaded = []
        
        for i, image in enumerate(images, 1):
            try:
                logger.info(f"Downloading {i}/{len(images)}: {image.camera}")
                path = self.download_image(image, overwrite=overwrite)
                downloaded.append(path)
            except Exception as e:
                logger.error(f"Failed to download image {i}: {e}")
                continue
        
        logger.info(f"Successfully downloaded {len(downloaded)}/{len(images)} images")
        return downloaded

    def get_manifest(self) -> Dict:
        """Get rover mission manifest with available data"""
        data = self._make_request('manifests/perseverance')
        return data.get('photo_manifest', {})


# Convenience functions
def fetch_latest_images(
    api_key: Optional[str] = None,
    camera: str = 'NAVCAM_LEFT',
    limit: int = 10,
    download: bool = True,
    cache_dir: str = "data/raw/perseverance"
) -> List[MarsImage]:
    """
    Quick function to fetch latest images
    
    Args:
        api_key: NASA API key
        camera: Camera name
        limit: Number of images
        download: Whether to download images
        cache_dir: Cache directory
        
    Returns:
        List of MarsImage objects
    """
    client = PerseveranceAPIClient(api_key=api_key, cache_dir=cache_dir)
    images = client.get_latest_photos(camera=camera, limit=limit)
    
    if download and images:
        client.download_multiple(images)
    
    return images


def fetch_images_by_sol(
    sol: int,
    api_key: Optional[str] = None,
    camera: Optional[str] = None,
    download: bool = True,
    max_downloads: Optional[int] = None,
    cache_dir: str = "data/raw/perseverance"
) -> List[MarsImage]:
    """
    Quick function to fetch images from specific sol
    
    Args:
        sol: Martian sol number
        api_key: NASA API key
        camera: Camera name (None = all)
        download: Whether to download images
        max_downloads: Maximum downloads
        cache_dir: Cache directory
        
    Returns:
        List of MarsImage objects
    """
    client = PerseveranceAPIClient(api_key=api_key, cache_dir=cache_dir)
    images = client.get_photos_by_sol(sol=sol, camera=camera)
    
    if download and images:
        client.download_multiple(images, max_downloads=max_downloads)
    
    return images


# Testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example 1: Get latest navigation camera images
    print("=" * 60)
    print("Fetching latest NAVCAM images...")
    print("=" * 60)
    
    images = fetch_latest_images(
        camera='NAVCAM_LEFT',
        limit=5,
        download=True
    )
    
    print(f"\nFound {len(images)} images:")
    for img in images:
        print(f"  Sol {img.sol} - {img.earth_date} - {img.camera}")
    
    # Example 2: Get images from specific sol
    print("\n" + "=" * 60)
    print("Fetching images from Sol 100...")
    print("=" * 60)
    
    sol_images = fetch_images_by_sol(
        sol=100,
        camera='NAVCAM_LEFT',
        download=False,
        max_downloads=3
    )
    
    print(f"\nFound {len(sol_images)} images from Sol 100")
    
    # Example 3: Get mission manifest
    print("\n" + "=" * 60)
    print("Mission Manifest:")
    print("=" * 60)
    
    client = PerseveranceAPIClient()
    manifest = client.get_manifest()
    
    print(f"  Landing Date: {manifest.get('landing_date')}")
    print(f"  Max Sol: {manifest.get('max_sol')}")
    print(f"  Max Earth Date: {manifest.get('max_date')}")
    print(f"  Total Photos: {manifest.get('total_photos')}")