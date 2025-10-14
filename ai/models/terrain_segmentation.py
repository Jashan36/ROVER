"""
Terrain segmentation model wrapper.
This file is a light wrapper around the UNet defined in `ai/models/unet.py`.
"""
from pathlib import Path
import torch

from ai.models.unet import UNet

class TerrainSegmentation:
    def __init__(self, weights_path=None, device='cpu'):
        self.device = device
        self.model = UNet(in_channels=3, out_channels=1).to(self.device)
        if weights_path:
            self.load(weights_path)

    def load(self, path):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)
        self.model.load_state_dict(torch.load(str(p), map_location=self.device))
        self.model.eval()

    def predict(self, img_tensor):
        with torch.no_grad():
            out = self.model(img_tensor.to(self.device))
            return out
