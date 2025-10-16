"""
Digital Elevation Model (DEM) Processor
Generates Mars terrain heightmaps for Gazebo simulation
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
import logging
import cv2

logger = logging.getLogger(__name__)


class DEMProcessor:
    """
    Processor for creating Digital Elevation Models (DEMs)
    """
    
    def __init__(self, output_dir: str = "data/raw/dem"):
        """
        Initialize DEM processor
        
        Args:
            output_dir: Directory to save generated DEMs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DEMProcessor initialized: {output_dir}")

    def create_crater_profile(
        self,
        size: Tuple[int, int] = (1024, 1024),
        crater_radius_km: float = 25.0,
        crater_depth_km: float = 0.5,
        rim_height_km: float = 0.1,
        noise_amplitude_km: float = 0.02,
        random_seed: int = 42
    ) -> np.ndarray:
        """
        Create crater profile similar to Jezero Crater
        
        Args:
            size: Output size (height, width)
            crater_radius_km: Crater radius in kilometers
            crater_depth_km: Maximum crater depth in kilometers
            rim_height_km: Crater rim elevation in kilometers
            noise_amplitude_km: Random noise amplitude in kilometers
            random_seed: Random seed for reproducibility
            
        Returns:
            Elevation array in kilometers
        """
        np.random.seed(random_seed)
        
        height, width = size
        
        # Create coordinate grids (in kilometers)
        extent_km = crater_radius_km * 2.5  # Total extent
        x = np.linspace(-extent_km/2, extent_km/2, width)
        y = np.linspace(-extent_km/2, extent_km/2, height)
        X, Y = np.meshgrid(x, y)
        
        # Radial distance from center
        R = np.sqrt(X**2 + Y**2)
        
        # Initialize elevation
        Z = np.zeros_like(R)
        
        # Crater bowl (cosine profile)
        crater_mask = R < crater_radius_km
        Z[crater_mask] = -crater_depth_km * (
            0.5 * (1 + np.cos(np.pi * R[crater_mask] / crater_radius_km))
        )
        
        # Crater rim (raised edge)
        rim_inner = crater_radius_km * 0.9
        rim_outer = crater_radius_km * 1.1
        rim_mask = (R >= rim_inner) & (R <= rim_outer)
        
        rim_profile = rim_height_km * np.exp(
            -((R[rim_mask] - crater_radius_km) ** 2) / (0.05 * crater_radius_km) ** 2
        )
        Z[rim_mask] += rim_profile
        
        # Add realistic noise (fractal-like)
        for scale in [1.0, 0.5, 0.25, 0.125]:
            freq = 10.0 / scale
            noise = noise_amplitude_km * scale * np.random.randn(height, width)
            noise = cv2.GaussianBlur(noise.astype(np.float32), (0, 0), sigmaX=width/(freq*2))
            Z += noise
        
        # Add some small craters
        n_small_craters = 20
        for _ in range(n_small_craters):
            cx = np.random.uniform(-extent_km/3, extent_km/3)
            cy = np.random.uniform(-extent_km/3, extent_km/3)
            cr = np.random.uniform(0.5, 3.0)  # Small crater radius
            cd = np.random.uniform(0.05, 0.15)  # Small crater depth
            
            crater_dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
            small_crater_mask = crater_dist < cr
            
            Z[small_crater_mask] -= cd * (
                0.5 * (1 + np.cos(np.pi * crater_dist[small_crater_mask] / cr))
            )
        
        # Add rocks (small elevation bumps)
        n_rocks = 50
        for _ in range(n_rocks):
            rx = np.random.uniform(-extent_km/2, extent_km/2)
            ry = np.random.uniform(-extent_km/2, extent_km/2)
            rsize = np.random.uniform(0.1, 0.5)
            rheight = np.random.uniform(0.005, 0.02)
            
            rock_dist = np.sqrt((X - rx)**2 + (Y - ry)**2)
            rock_mask = rock_dist < rsize
            
            Z[rock_mask] += rheight * np.exp(-(rock_dist[rock_mask] / rsize) ** 2)
        
        logger.info(f"Generated crater profile: {size}, range=[{Z.min():.3f}, {Z.max():.3f}] km")
        
        return Z

    def normalize_to_16bit(
        self,
        elevation: np.ndarray,
        z_min: Optional[float] = None,
        z_max: Optional[float] = None
    ) -> np.ndarray:
        """
        Normalize elevation to 16-bit range
        
        Args:
            elevation: Elevation array in physical units
            z_min: Minimum elevation (None = use data min)
            z_max: Maximum elevation (None = use data max)
            
        Returns:
            16-bit unsigned integer array
        """
        if z_min is None:
            z_min = elevation.min()
        if z_max is None:
            z_max = elevation.max()
        
        # Normalize to [0, 1]
        normalized = (elevation - z_min) / (z_max - z_min)
        
        # Scale to 16-bit range
        scaled = (normalized * 65535).clip(0, 65535).astype(np.uint16)
        
        logger.debug(f"Normalized: [{z_min:.3f}, {z_max:.3f}] -> [0, 65535]")
        
        return scaled

    def save_heightmap(
        self,
        elevation: np.ndarray,
        filename: str,
        z_range_meters: Tuple[float, float] = (0.0, 10.0)
    ) -> Path:
        """
        Save heightmap as 16-bit PNG
        
        Args:
            elevation: Elevation array in kilometers
            filename: Output filename
            z_range_meters: Desired elevation range in meters for simulation
            
        Returns:
            Path to saved file
        """
        filepath = self.output_dir / filename
        
        # Convert km to meters
        elevation_m = elevation * 1000.0
        
        # Normalize to specified range
        heightmap_16bit = self.normalize_to_16bit(
            elevation_m,
            z_min=z_range_meters[0],
            z_max=z_range_meters[1]
        )
        
        # Save as PNG
        img = Image.fromarray(heightmap_16bit, mode='I;16')
        img.save(filepath)
        
        # Save metadata
        metadata = {
            'filename': filename,
            'size': elevation.shape,
            'z_range_km': [float(elevation.min()), float(elevation.max())],
            'z_range_meters': list(z_range_meters),
            'dtype': 'uint16',
            'encoding': '16-bit grayscale PNG'
        }
        
        import json
        metadata_path = filepath.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved heightmap: {filepath}")
        logger.info(f"  Size: {elevation.shape}")
        logger.info(f"  Range: {z_range_meters[0]:.1f}m to {z_range_meters[1]:.1f}m")
        
        return filepath

    def create_texture_map(
        self,
        elevation: np.ndarray,
        colormap: str = 'mars'
    ) -> np.ndarray:
        """
        Create texture map from elevation
        
        Args:
            elevation: Elevation array
            colormap: Color scheme ('mars', 'grayscale', 'terrain')
            
        Returns:
            RGB texture array (H, W, 3)
        """
        # Normalize elevation
        normalized = (elevation - elevation.min()) / (elevation.max() - elevation.min())
        
        if colormap == 'mars':
            # Mars-like reddish-brown colors
            r = (180 + normalized * 75).astype(np.uint8)
            g = (80 + normalized * 60).astype(np.uint8)
            b = (30 + normalized * 40).astype(np.uint8)
            
        elif colormap == 'grayscale':
            gray = (normalized * 255).astype(np.uint8)
            r = g = b = gray
            
        elif colormap == 'terrain':
            # Import matplotlib for terrain colormap
            try:
                import matplotlib.cm as cm
                terrain = cm.get_cmap('terrain')
                rgba = terrain(normalized)
                r = (rgba[:, :, 0] * 255).astype(np.uint8)
                g = (rgba[:, :, 1] * 255).astype(np.uint8)
                b = (rgba[:, :, 2] * 255).astype(np.uint8)
            except ImportError:
                logger.warning("matplotlib not available, using grayscale")
                gray = (normalized * 255).astype(np.uint8)
                r = g = b = gray
        
        else:
            raise ValueError(f"Unknown colormap: {colormap}")
        
        texture = np.stack([r, g, b], axis=-1)
        return texture

    def save_texture(
        self,
        texture: np.ndarray,
        filename: str
    ) -> Path:
        """Save texture map as image"""
        filepath = self.output_dir / filename
        
        img = Image.fromarray(texture, mode='RGB')
        img.save(filepath)
        
        logger.info(f"Saved texture: {filepath}")
        return filepath


# Convenience function
def generate_mars_heightmap(
    output_dir: str = "sim/rover_description/worlds",
    size: Tuple[int, int] = (1024, 1024),
    z_range_meters: Tuple[float, float] = (0.0, 10.0),
    with_texture: bool = True
) -> Tuple[Path, Optional[Path]]:
    """
    Generate Mars terrain heightmap for Gazebo
    
    Args:
        output_dir: Output directory
        size: Heightmap size
        z_range_meters: Elevation range in meters
        with_texture: Also generate texture map
        
    Returns:
        Tuple of (heightmap_path, texture_path)
    """
    processor = DEMProcessor(output_dir=output_dir)
    
    # Create crater profile
    elevation = processor.create_crater_profile(size=size)
    
    # Save heightmap
    heightmap_path = processor.save_heightmap(
        elevation,
        filename='mars_heightmap.png',
        z_range_meters=z_range_meters
    )
    
    # Generate and save texture
    texture_path = None
    if with_texture:
        texture = processor.create_texture_map(elevation, colormap='mars')
        texture_path = processor.save_texture(texture, filename='mars_texture.png')
    
    return heightmap_path, texture_path


def create_crater_profile(
    size: Tuple[int, int] = (1024, 1024),
    **kwargs
) -> np.ndarray:
    """
    Quick function to create crater elevation profile
    
    Args:
        size: Output size
        **kwargs: Additional parameters for create_crater_profile
        
    Returns:
        Elevation array in kilometers
    """
    processor = DEMProcessor()
    return processor.create_crater_profile(size=size, **kwargs)


# Testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("DEM Processor - Generating Mars Terrain")
    print("=" * 60)
    
    # Generate heightmap
    print("\nGenerating crater profile...")
    heightmap_path, texture_path = generate_mars_heightmap(
        output_dir="data/raw/dem",
        size=(1024, 1024),
        z_range_meters=(0.0, 10.0),
        with_texture=True
    )
    
    print(f"\nGenerated files:")
    print(f"  Heightmap: {heightmap_path}")
    print(f"  Texture: {texture_path}")
    
    # Visualize
    print("\nTo visualize:")
    print("  from PIL import Image")
    print(f"  img = Image.open('{heightmap_path}')")
    print("  img.show()")