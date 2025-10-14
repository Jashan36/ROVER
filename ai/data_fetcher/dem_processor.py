"""
Process USGS DEM files into heightmaps usable by Gazebo.
This is a stub demonstrating how to read common DEM formats using rasterio (optional dependency).
"""
from pathlib import Path

try:
    import rasterio
except Exception:
    rasterio = None


def dem_to_heightmap(dem_path, out_png, scale=1.0):
    if rasterio is None:
        raise RuntimeError('rasterio is required for DEM processing')
    with rasterio.open(dem_path) as src:
        arr = src.read(1)
    # Normalize and write as PNG heightmap
    import numpy as np
    from PIL import Image
    a = (arr - arr.min()) / (arr.max() - arr.min())
    a = (a * 255).astype('uint8')
    Image.fromarray(a).save(out_png)
    return out_png
