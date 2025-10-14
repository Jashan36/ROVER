"""
Model Cache
Efficient model loading and caching to speed up initialization
"""

import torch
from pathlib import Path
from typing import Dict, Optional, Any
import hashlib
import pickle
import logging
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for model cache"""
    cache_dir: str = "data/cache/model_cache"
    enable_caching: bool = True
    cache_ttl_seconds: int = 86400  # 24 hours
    verify_checksum: bool = True


class ModelCache:
    """
    Cache for loaded models to speed up initialization
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize model cache
        
        Args:
            config: Cache configuration
        """
        self.config = config or CacheConfig()
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._memory_cache: Dict[str, Any] = {}
        
        logger.info(f"ModelCache initialized: {self.cache_dir}")

    def get_cache_key(self, model_path: Path, **kwargs) -> str:
        """
        Generate cache key for model
        
        Args:
            model_path: Path to model file
            **kwargs: Additional parameters affecting cache
            
        Returns:
            Cache key string
        """
        # Create string from path and parameters
        key_str = f"{model_path}_{str(sorted(kwargs.items()))}"
        
        # Hash for consistent key length
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        
        return key_hash

    def load_model(
        self,
        model_path: Path,
        model_class: type,
        device: str = 'cpu',
        **kwargs
    ) -> Any:
        """
        Load model with caching
        
        Args:
            model_path: Path to model weights
            model_class: Model class to instantiate
            device: Device for model
            **kwargs: Additional model parameters
            
        Returns:
            Loaded model
        """
        cache_key = self.get_cache_key(model_path, device=device, **kwargs)
        
        # Check memory cache first
        if cache_key in self._memory_cache:
            logger.debug(f"Model loaded from memory cache: {cache_key[:8]}")
            return self._memory_cache[cache_key]
        
        # Check disk cache
        if self.config.enable_caching:
            cached_model = self._load_from_disk(cache_key, device)
            if cached_model is not None:
                self._memory_cache[cache_key] = cached_model
                return cached_model
        
        # Load model from scratch
        logger.info(f"Loading model from {model_path}")
        model = model_class(**kwargs)
        
        if model_path and model_path.exists():
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        # Cache model
        self._memory_cache[cache_key] = model
        
        if self.config.enable_caching:
            self._save_to_disk(cache_key, model)
        
        return model

    def _load_from_disk(self, cache_key: str, device: str) -> Optional[Any]:
        """Load model from disk cache"""
        cache_path = self.cache_dir / f"{cache_key}.pt"
        metadata_path = self.cache_dir / f"{cache_key}.json"
        
        if not cache_path.exists() or not metadata_path.exists():
            return None
        
        try:
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Check TTL
            import time
            age = time.time() - metadata['timestamp']
            if age > self.config.cache_ttl_seconds:
                logger.debug(f"Cache expired: {cache_key[:8]}")
                return None
            
            # Load model
            model = torch.load(cache_path, map_location=device)
            logger.info(f"Model loaded from disk cache: {cache_key[:8]}")
            
            return model
            
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
            return None

    def _save_to_disk(self, cache_key: str, model: Any):
        """Save model to disk cache"""
        cache_path = self.cache_dir / f"{cache_key}.pt"
        metadata_path = self.cache_dir / f"{cache_key}.json"
        
        try:
            # Save model
            torch.save(model, cache_path)
            
            # Save metadata
            import time
            metadata = {
                'cache_key': cache_key,
                'timestamp': time.time(),
                'device': str(next(model.parameters()).device)
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            logger.debug(f"Model saved to cache: {cache_key[:8]}")
            
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")

    def clear_memory_cache(self):
        """Clear in-memory cache"""
        self._memory_cache.clear()
        logger.info("Memory cache cleared")

    def clear_disk_cache(self):
        """Clear disk cache"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True)
        logger.info("Disk cache cleared")

    def get_cache_size(self) -> Dict[str, int]:
        """Get cache size information"""
        disk_files = list(self.cache_dir.glob("*.pt"))
        disk_size = sum(f.stat().st_size for f in disk_files)
        
        return {
            'memory_entries': len(self._memory_cache),
            'disk_files': len(disk_files),
            'disk_size_mb': disk_size / (1024 * 1024)
        }


# Testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Model Cache Test")
    print("=" * 60)
    
    cache = ModelCache()
    
    # Check size
    size_info = cache.get_cache_size()
    print(f"\nCache Size:")
    print(f"  Memory entries: {size_info['memory_entries']}")
    print(f"  Disk files: {size_info['disk_files']}")
    print(f"  Disk size: {size_info['disk_size_mb']:.2f} MB")