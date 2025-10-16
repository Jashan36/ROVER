#!/usr/bin/env python3
"""
Generate a Mars-like heightmap for Gazebo simulation
Creates a 16-bit grayscale PNG with crater-like terrain
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def generate_mars_heightmap():
    # Create coordinate grid
    width, height = 1024, 1024
    x = np.linspace(-25, 25, width)
    y = np.linspace(-25, 25, height)
    X, Y = np.meshgrid(x, y)
    
    # Generate main crater (Jezero-like)
    R = np.sqrt(X**2 + Y**2)
    crater_rim = 8.0  # km
    crater_depth = 0.3  # km
    
    # Crater profile (cosine-based)
    Z = np.zeros_like(R)
    crater_mask = R < crater_rim
    Z[crater_mask] = -crater_depth * np.cos(np.pi * R[crater_mask] / crater_rim)
    
    # Add smaller craters
    for i in range(20):
        cx = np.random.uniform(-20, 20)
        cy = np.random.uniform(-20, 20)
        cr = np.random.uniform(0.5, 3.0)
        small_R = np.sqrt((X - cx)**2 + (Y - cy)**2)
        small_mask = small_R < cr
        small_depth = np.random.uniform(0.05, 0.15)
        Z[small_mask] -= small_depth * np.cos(np.pi * small_R[small_mask] / cr)
    
    # Add realistic noise (small craters, rocks)
    noise = 0.02 * np.random.randn(*Z.shape)
    Z += noise
    
    # Normalize to 0-10 meter range for simulation
    Z_normalized = (Z - Z.min()) / (Z.max() - Z.min()) * 10.0
    
    # Convert to 16-bit (0-65535)
    Z_16bit = (Z_normalized / 10.0 * 65535).astype(np.uint16)
    
    # Save as 16-bit PNG
    img = Image.fromarray(Z_16bit)
    img.save('mars_heightmap.png')
    
    print("✅ Mars heightmap generated: mars_heightmap.png")
    print(f"📊 Elevation range: {Z_normalized.min():.2f}m to {Z_normalized.max():.2f}m")

if __name__ == '__main__':
    generate_mars_heightmap()