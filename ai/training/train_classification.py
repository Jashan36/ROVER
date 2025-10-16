"""
Training script for Mars image classification using CSV metadata.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .dataset import get_data_loaders, MarsClassificationDataset
from .augmentations import (
    get_classification_training_augmentations,
    get_classification_validation_augmentations,
)
from ai.models.mars_classifier import MarsClassifier

logger = logging.getLogger(__name__)


@dataclass
class ClassificationTrainingConfig:
    # Data
    data_root: str = "data/raw/mars_dataset/splits"
    metadata_csv: str = "data/raw/mars_dataset/mars_rover_dataset.csv"
    label_column: str = "camera_name"
    image_size: tuple = (512, 512)
    batch_size: int = 32
    num_workers: int = 4

    # Training
    num_epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adam"

    # Scheduler
    use_scheduler: bool = True
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    min_lr: float = 1e-6

    # Checkpointing
    checkpoint_dir: str = "ai/models/weights"
    checkpoint_name: str = "mars_classifier"
    save_best_only: bool = True

    # Logging
    log_dir: str = "logs/training_classification"
    log_frequency: int = 10

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Misc
    dropout: float = 0.2
    pretrained_backbone: bool = True
    early_stopping_patience: int = 10

    def to_dict(self) -> Dict:
        config = asdict(self)
        config["image_size"] = list(config["image_size"])
        return config


class ClassificationTrainer:
    def __init__(self, config: ClassificationTrainingConfig):
        self.config = config

        logger.info("Initialising classification trainer...")
        self._log_config()

        # Directories
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(self.config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Data loaders
        self.train_loader, self.val_loader, self.test_loader = self._build_data_loaders()

        # Model
        num_classes = self.train_loader.dataset.num_classes  # type: ignore[attr-defined]
        self.model = MarsClassifier(
            num_classes=num_classes,
            pretrained=self.config.pretrained_backbone,
            dropout=self.config.dropout,
        ).to(self.config.device)

        # Criterion
        class_weights = None
        if hasattr(self.train_loader.dataset, "get_class_weights"):
            class_weights = self.train_loader.dataset.get_class_weights()  # type: ignore[attr-defined]
        if class_weights is not None:
            class_weights = class_weights.to(self.config.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer
        if self.config.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer '{self.config.optimizer}'")

        # Scheduler
        self.scheduler = None
        if self.config.use_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                patience=self.config.scheduler_patience,
                factor=self.config.scheduler_factor,
                min_lr=self.config.min_lr,
)

        # Logging
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Training state
        self.current_epoch = 0
        self.best_val_accuracy = 0.0
        self.epochs_without_improvement = 0

        logger.info("Classification trainer initialised with %d classes", num_classes)

    def _build_data_loaders(self) -> tuple:
        train_aug = get_classification_training_augmentations(
            image_size=self.config.image_size,
            intensity="medium",
        )
        val_aug = get_classification_validation_augmentations(
            image_size=self.config.image_size
        )

        return get_data_loaders(
            root_dir=self.config.data_root,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            target_size=self.config.image_size,
            train_transform=train_aug,
            val_transform=val_aug,
            dataset_type="classification",
            metadata_csv=self.config.metadata_csv,
            label_column=self.config.label_column,
        )

    def train(self):
        logger.info("=" * 60)
        logger.info("Starting classification training")
        logger.info("=" * 60)

        start_time = time.time()

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            train_metrics = self._train_one_epoch()
            val_metrics = self._validate()

            logger.info(
                "Epoch %d/%d - Train loss: %.4f, acc: %.4f - Val loss: %.4f, acc: %.4f",
                epoch + 1,
                self.config.num_epochs,
                train_metrics["loss"],
                train_metrics["accuracy"],
                val_metrics["loss"],
                val_metrics["accuracy"],
            )

            self.writer.add_scalar("train/loss", train_metrics["loss"], epoch)
            self.writer.add_scalar("train/accuracy", train_metrics["accuracy"], epoch)
            self.writer.add_scalar("val/loss", val_metrics["loss"], epoch)
            self.writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
            self.writer.add_scalar(
                "train/learning_rate", self.optimizer.param_groups[0]["lr"], epoch
            )

            if self.scheduler:
                self.scheduler.step(val_metrics["accuracy"])

            improved = val_metrics["accuracy"] > self.best_val_accuracy
            if improved:
                self.best_val_accuracy = val_metrics["accuracy"]
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            self._save_checkpoint(is_best=improved)

            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                logger.info(
                    "Early stopping triggered after %d epochs", epoch + 1
                )
                break

        total_time = time.time() - start_time
        logger.info("Training finished in %.2f minutes", total_time / 60.0)
        logger.info("Best validation accuracy: %.4f", self.best_val_accuracy)
        self.writer.close()

    def _train_one_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [train]")
        for step, (images, labels) in enumerate(pbar):
            images = images.to(self.config.device)
            labels = labels.to(self.config.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if step % self.config.log_frequency == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                global_step = self.current_epoch * len(self.train_loader) + step
                self.writer.add_scalar("train/batch_loss", loss.item(), global_step)

        avg_loss = total_loss / total
        accuracy = correct / total if total else 0.0
        return {"loss": avg_loss, "accuracy": accuracy}

    def _validate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.config.device)
                labels = labels.to(self.config.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / total if total else 0.0
        accuracy = correct / total if total else 0.0
        return {"loss": avg_loss, "accuracy": accuracy}

    def _save_checkpoint(self, is_best: bool = False):
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_accuracy": self.best_val_accuracy,
            "config": self.config.to_dict(),
        }
        checkpoint_path = self.checkpoint_dir / f"{self.config.checkpoint_name}_latest.pth"
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = self.checkpoint_dir / f"{self.config.checkpoint_name}_best.pth"
            torch.save(checkpoint, best_path)
            logger.info("Saved new best model to %s", best_path)

    def _log_config(self):
        for key, value in self.config.to_dict().items():
            logger.info("  %s: %s", key, value)


def train_classification_model(
    data_root: str = "data/raw/mars_dataset/splits",
    metadata_csv: str = "data/raw/mars_dataset/mars_rover_dataset.csv",
    label_column: str = "camera_name",
    num_epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str = "auto",
    num_workers: Optional[int] = None,
) -> ClassificationTrainer:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = ClassificationTrainingConfig(
        data_root=data_root,
        metadata_csv=metadata_csv,
        label_column=label_column,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_workers=num_workers if num_workers is not None else ClassificationTrainingConfig().num_workers,
        device=device,
    )

    trainer = ClassificationTrainer(config)
    trainer.train()
    return trainer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Mars image classifier.")
    parser.add_argument("--data-root", default="data/raw/mars_dataset/splits")
    parser.add_argument("--metadata-csv", default="data/raw/mars_dataset/mars_rover_dataset.csv")
    parser.add_argument("--label-column", default="camera_name")
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--num-workers", type=int, default=None)

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    trainer = train_classification_model(
        data_root=args.data_root,
        metadata_csv=args.metadata_csv,
        label_column=args.label_column,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        num_workers=args.num_workers if args.num_workers is not None else ClassificationTrainingConfig().num_workers,
    )
    print("Training complete. Best accuracy:", trainer.best_val_accuracy)
