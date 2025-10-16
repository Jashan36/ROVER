"""
Simple Mars terrain image classifier built on top of torchvision backbones.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)


_RESNET_WEIGHTS = {
    "resnet18": getattr(models, "ResNet18_Weights", None),
    "resnet34": getattr(models, "ResNet34_Weights", None),
}


class MarsClassifier(nn.Module):
    """
    Generic classifier wrapper using torchvision models with a custom head.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: Literal["resnet18", "resnet34"] = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes

        if backbone not in ("resnet18", "resnet34"):
            raise ValueError(f"Unsupported backbone '{backbone}'")

        weights_enum = _RESNET_WEIGHTS.get(backbone)
        weights = None
        if pretrained and weights_enum is not None:
            weights = weights_enum.DEFAULT

        backbone_ctor = getattr(models, backbone)
        self.backbone = backbone_ctor(weights=weights)

        in_features = self.backbone.fc.in_features  # type: ignore[attr-defined]
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

        logger.info(
            "MarsClassifier initialised with backbone=%s pretrained=%s num_classes=%d",
            backbone,
            pretrained,
            num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
