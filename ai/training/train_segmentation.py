"""
Training Script for Terrain Segmentation Model
Complete training pipeline with checkpointing, logging, and evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass, asdict
import logging
import json
import time
from tqdm import tqdm
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.terrain_segmentation import UNet
from training.dataset import get_data_loaders
from training.augmentations import get_training_augmentations, get_validation_augmentations

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Data
    data_root: str = "data/raw/ai4mars"
    image_size: tuple = (512, 512)
    batch_size: int = 16
    num_workers: int = 4
    
    # Model
    n_channels: int = 3
    n_classes: int = 5
    
    # Training
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    
    # Optimizer
    optimizer: str = 'adam'  # 'adam' or 'sgd'
    momentum: float = 0.9  # for SGD
    
    # Learning rate schedule
    use_scheduler: bool = True
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    min_lr: float = 1e-6
    
    # Loss
    use_class_weights: bool = True
    label_smoothing: float = 0.1
    
    # Checkpointing
    checkpoint_dir: str = "ai/models/weights"
    save_frequency: int = 10
    save_best_only: bool = True
    
    # Logging
    log_dir: str = "logs/training"
    log_frequency: int = 10
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Mixed precision
    use_amp: bool = False  # Automatic Mixed Precision
    
    # Early stopping
    early_stopping_patience: int = 15
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        config_dict = asdict(self)
        config_dict['image_size'] = list(config_dict['image_size'])
        return config_dict


class Trainer:
    """
    Training manager for terrain segmentation
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        Initialize trainer
        
        Args:
            config: Training configuration
        """
        self.config = config or TrainingConfig()
        
        logger.info("Initializing Trainer...")
        self._log_config()
        
        # Setup directories
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(self.config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = self._build_model()
        
        # Initialize data loaders
        self.train_loader, self.val_loader, self.test_loader = self._build_data_loaders()
        
        # Initialize loss
        self.criterion = self._build_criterion()
        
        # Initialize optimizer
        self.optimizer = self._build_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._build_scheduler()
        
        # Initialize tensorboard
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_miou = 0.0
        self.epochs_without_improvement = 0
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.config.use_amp else None
        
        logger.info("Trainer initialized successfully")

    def _build_model(self) -> nn.Module:
        """Build model"""
        model = UNet(
            n_channels=self.config.n_channels,
            n_classes=self.config.n_classes
        )
        model.to(self.config.device)
        
        logger.info(f"Model: U-Net with {self.config.n_classes} classes")
        logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model

    def _build_data_loaders(self) -> tuple:
        """Build data loaders"""
        train_aug = get_training_augmentations(
            image_size=self.config.image_size,
            intensity='medium'
        )
        
        val_aug = get_validation_augmentations(
            image_size=self.config.image_size
        )
        
        train_loader, val_loader, test_loader = get_data_loaders(
            root_dir=self.config.data_root,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            target_size=self.config.image_size,
            train_transform=train_aug,
            val_transform=val_aug
        )
        
        logger.info(f"Data loaders created:")
        logger.info(f"  Train batches: {len(train_loader)}")
        logger.info(f"  Val batches: {len(val_loader)}")
        if test_loader:
            logger.info(f"  Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader

    def _build_criterion(self) -> nn.Module:
        """Build loss function"""
        if self.config.use_class_weights:
            # Calculate class weights from training data
            logger.info("Calculating class weights...")
            class_weights = self.train_loader.dataset.get_class_weights()
            class_weights = class_weights.to(self.config.device)
            logger.info(f"Class weights: {class_weights.cpu().numpy()}")
        else:
            class_weights = None
        
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=self.config.label_smoothing
        )
        
        return criterion

    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer"""
        if self.config.optimizer.lower() == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        logger.info(f"Optimizer: {self.config.optimizer.upper()}")
        
        return optimizer

    def _build_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Build learning rate scheduler"""
        if not self.config.use_scheduler:
            return None
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.config.scheduler_patience,
            factor=self.config.scheduler_factor,
            min_lr=self.config.min_lr,
            verbose=True
        )
        
        logger.info("Learning rate scheduler: ReduceLROnPlateau")
        
        return scheduler

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.config.device)
            labels = labels.to(self.config.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.config.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Statistics
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Log to tensorboard
            if batch_idx % self.config.log_frequency == 0:
                global_step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('train/batch_loss', loss.item(), global_step)
        
        avg_loss = total_loss / total_samples
        
        return {'loss': avg_loss}

    def validate(self) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        
        # For mIoU calculation
        class_iou = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validating"):
                images = images.to(self.config.device)
                labels = labels.to(self.config.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Calculate IoU
                preds = torch.argmax(outputs, dim=1)
                batch_iou = self._calculate_miou(preds, labels)
                class_iou.append(batch_iou)
        
        avg_loss = total_loss / total_samples
        miou = np.mean(class_iou)
        
        return {
            'loss': avg_loss,
            'miou': miou
        }

    def _calculate_miou(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """Calculate mean Intersection over Union"""
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
        
        ious = []
        
        for cls in range(self.config.n_classes):
            pred_cls = (preds == cls)
            label_cls = (labels == cls)
            
            intersection = (pred_cls & label_cls).sum()
            union = (pred_cls | label_cls).sum()
            
            if union > 0:
                ious.append(intersection / union)
        
        return np.mean(ious) if ious else 0.0

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_miou': self.best_miou,
            'config': self.config.to_dict()
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest
        latest_path = self.checkpoint_dir / 'terrain_unet_latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / 'terrain_unet_best.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"✓ Saved best model (mIoU: {self.best_miou:.4f})")
        
        # Save periodic
        if (self.current_epoch + 1) % self.config.save_frequency == 0:
            epoch_path = self.checkpoint_dir / f'terrain_unet_epoch_{self.current_epoch + 1}.pth'
            torch.save(checkpoint, epoch_path)

    def train(self):
        """Main training loop"""
        logger.info("=" * 60)
        logger.info("Starting Training")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Log metrics
            logger.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f} - "
                f"Val Loss: {val_metrics['loss']:.4f} - "
                f"Val mIoU: {val_metrics['miou']:.4f}"
            )
            
            self.writer.add_scalar('train/loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('val/loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('val/miou', val_metrics['miou'], epoch)
            self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step(val_metrics['loss'])
            
            # Check for improvement
            is_best = False
            if val_metrics['miou'] > self.best_miou:
                self.best_miou = val_metrics['miou']
                self.best_val_loss = val_metrics['loss']
                is_best = True
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        total_time = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info(f"Total time: {total_time / 3600:.2f} hours")
        logger.info(f"Best mIoU: {self.best_miou:.4f}")
        logger.info("=" * 60)
        
        self.writer.close()

    def _log_config(self):
        """Log configuration"""
        logger.info("Training Configuration:")
        for key, value in self.config.to_dict().items():
            logger.info(f"  {key}: {value}")


def train_model(
    data_root: str = "data/raw/ai4mars",
    num_epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    device: str = 'auto'
) -> Trainer:
    """
    Convenience function to train model
    
    Args:
        data_root: Path to AI4Mars data
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: 'cuda', 'cpu', or 'auto'
        
    Returns:
        Trained Trainer instance
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = TrainingConfig(
        data_root=data_root,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device
    )
    
    trainer = Trainer(config)
    trainer.train()
    
    return trainer


# Main execution
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("Mars Rover Terrain Segmentation Training")
    print("=" * 60)
    
    # Train model
    trainer = train_model(
        data_root="data/raw/ai4mars",
        num_epochs=100,
        batch_size=16,
        learning_rate=0.001
    )
    
    print("\nTraining complete! Model saved to ai/models/weights/")