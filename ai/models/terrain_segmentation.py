"""
Terrain Segmentation Model using U-Net Architecture
Optimized for Mars terrain classification with real-time inference capability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DoubleConv(nn.Module):
    """Double convolution block with BatchNorm and ReLU"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """
    U-Net for Mars terrain segmentation
    Classes: soil, bedrock, sand, big_rock, background
    """
    def __init__(self, n_channels: int = 3, n_classes: int = 5, bilinear: bool = True, base_channels: int = 32):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Channel configuration (lighter network by default)
        c1, c2, c3, c4, c5 = base_channels, base_channels*2, base_channels*4, base_channels*8, base_channels*16

        # Encoder
        self.inc = DoubleConv(n_channels, c1)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c1, c2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c2, c3))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c3, c4))
        
        factor = 2 if bilinear else 1
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c4, c5 // factor))

        # Decoder
        self.up1 = nn.ConvTranspose2d(c5 // factor, c4 // factor, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(c4 + (c4 // factor), c4 // factor)
        
        self.up2 = nn.ConvTranspose2d(c4 // factor, c3 // factor, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(c3 + (c3 // factor), c3 // factor)
        
        self.up3 = nn.ConvTranspose2d(c3 // factor, c2 // factor, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(c2 + (c2 // factor), c2 // factor)
        
        self.up4 = nn.ConvTranspose2d(c2 // factor, c1, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(c1 + c1, c1)
        
        # Output layer
        self.outc = nn.Conv2d(c1, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with skip connections
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv4(x)

        logits = self.outc(x)
        return logits


class TerrainSegmentationModel:
    """
    Production-ready terrain segmentation with model management
    """
    CLASS_NAMES = ['soil', 'bedrock', 'sand', 'big_rock', 'background']
    CLASS_COLORS = np.array([
        [139, 69, 19],   # soil - brown
        [128, 128, 128], # bedrock - gray
        [255, 228, 181], # sand - tan
        [105, 105, 105], # big_rock - dark gray
        [0, 0, 0]        # background - black
    ], dtype=np.uint8)

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        input_size: Tuple[int, int] = (512, 512)
    ):
        self.device = torch.device(device)
        self.input_size = input_size
        
        logger.info(f"Initializing TerrainSegmentationModel on {self.device}")
        
        # Initialize model (extra-light U-Net for CPU-friendly inference)
        self.model = UNet(n_channels=3, n_classes=len(self.CLASS_NAMES), base_channels=8)
        self.model.to(self.device)
        
        # Load pretrained weights if available
        if model_path and Path(model_path).exists():
            self.load_weights(model_path)
        else:
            logger.warning("No pretrained weights loaded - using random initialization")
        
        self.model.eval()
        
        # Normalization parameters (ImageNet stats)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def load_weights(self, path: Path):
        """Load pretrained model weights"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded model weights from {path}")
                if 'epoch' in checkpoint:
                    logger.info(f"Model trained for {checkpoint['epoch']} epochs")
            else:
                self.model.load_state_dict(checkpoint)
                logger.info(f"Loaded model weights from {path}")
                
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")
            raise

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for inference
        Args:
            image: RGB image (H, W, 3) in range [0, 255]
        Returns:
            Normalized tensor (1, 3, H, W)
        """
        # Validate input
        if image is None or not hasattr(image, 'shape'):
            raise ValueError("Invalid image: None")
        if image.size == 0:
            raise ValueError("Invalid image: empty")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Invalid image shape: expected HxWx3")
        if image.shape[0] < 16 or image.shape[1] < 16:
            raise ValueError("Invalid image size: too small")
        # Resize
        if image.shape[:2] != self.input_size:
            import cv2
            image = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert to tensor and normalize
        tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        tensor = (tensor - self.mean) / self.std
        
        return tensor

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        return_confidence: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Perform inference on input image
        Args:
            image: RGB image (H, W, 3) in range [0, 255]
            return_confidence: Whether to return confidence scores
        Returns:
            Dictionary with segmentation results
        """
        original_shape = image.shape[:2]
        
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Inference
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)
        
        # Get predictions
        confidence, pred_classes = torch.max(probs, dim=1)
        
        # Convert to numpy
        pred_classes = pred_classes.squeeze().cpu().numpy().astype(np.uint8)
        confidence = confidence.squeeze().cpu().numpy().astype(np.float32)
        
        # Resize back to original if needed
        if pred_classes.shape != original_shape:
            import cv2
            pred_classes = cv2.resize(
                pred_classes, 
                (original_shape[1], original_shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            confidence = cv2.resize(
                confidence,
                (original_shape[1], original_shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        
        # Create colored segmentation map
        segmentation_colored = self.CLASS_COLORS[pred_classes]
        
        result = {
            'classes': pred_classes,
            'segmentation_colored': segmentation_colored,
        }
        
        if return_confidence:
            result['confidence'] = confidence
        
        return result

    def create_overlay(
        self,
        image: np.ndarray,
        segmentation: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """Create overlay visualization"""
        overlay = (alpha * segmentation + (1 - alpha) * image).astype(np.uint8)
        return overlay

    @staticmethod
    def get_class_stats(pred_classes: np.ndarray) -> Dict[str, float]:
        """Calculate class distribution statistics"""
        unique, counts = np.unique(pred_classes, return_counts=True)
        total_pixels = pred_classes.size
        
        stats = {}
        for cls, count in zip(unique, counts):
            if cls < len(TerrainSegmentationModel.CLASS_NAMES):
                class_name = TerrainSegmentationModel.CLASS_NAMES[cls]
                stats[class_name] = float(count / total_pixels)
        
        return stats


# Testing function
if __name__ == "__main__":
    # Test model
    model = TerrainSegmentationModel()
    
    # Create dummy input
    test_image = np.random.randint(0, 255, (960, 1280, 3), dtype=np.uint8)
    
    # Run inference
    import time
    start = time.time()
    results = model.predict(test_image)
    end = time.time()
    
    logger.info(f"Inference time: {(end-start)*1000:.1f}ms")
    logger.info(f"Output shape: {results['classes'].shape}")
    logger.info(f"Class distribution: {model.get_class_stats(results['classes'])}")
