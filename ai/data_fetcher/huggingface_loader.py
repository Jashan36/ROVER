"""
HuggingFace dataset/model loader implementation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .auth_manager import AuthManager
from .base_loader import BaseDataLoader, DatasetMetadata, DownloadResult
from .dependency_manager import DependencyManager
from .universal_loader import register_loader

logger = logging.getLogger(__name__)


class HuggingFaceDataLoader(BaseDataLoader):
    """
    Loader that leverages ``datasets`` and ``huggingface_hub`` to access hosted
    datasets or models.
    """

    def __init__(self, metadata: DatasetMetadata, params: Dict[str, Any]) -> None:
        self.params = params
        super().__init__(
            metadata,
            cache_dir=params.get("cache_dir"),
            download_dir=params.get("download_dir"),
            force=params.get("force", False),
        )
        self.token = AuthManager.setup_credentials("huggingface", **params).get("token")

    def download(self) -> DownloadResult:
        DependencyManager.check("huggingface")
        from datasets import load_dataset  # type: ignore

        dataset_id = self.metadata.dataset_id
        config = self.params.get("config")
        split = self.params.get("split")
        cache_dir = str(self.resolve_cache_path("hf_cache"))

        logger.info("Fetching HuggingFace dataset %s", dataset_id)
        dataset = load_dataset(dataset_id, name=config, split=split, cache_dir=cache_dir, token=self.token)

        metadata = {
            "features": dataset.features if hasattr(dataset, "features") else None,
            "num_rows": len(dataset),
        }

        path = Path(cache_dir)
        return DownloadResult(paths=(path,), metadata=metadata, cached=False)

    def load(self, **kwargs: Any) -> Any:
        from datasets import load_dataset  # type: ignore

        dataset_id = self.metadata.dataset_id
        config = self.params.get("config")
        split = self.params.get("split")
        cache_dir = str(self.resolve_cache_path("hf_cache"))
        return load_dataset(dataset_id, name=config, split=split, cache_dir=cache_dir, token=self.token)

    def preprocess(self, **kwargs: Any) -> Any:
        return None

    def create_splits(self, **kwargs: Any) -> Dict[str, Any]:
        return {}

    def get_statistics(self) -> Dict[str, Any]:
        dataset = self.load()
        return {
            "num_rows": len(dataset),
            "columns": list(dataset.column_names),
        }


def _factory(metadata: DatasetMetadata, params: Dict[str, Any]) -> BaseDataLoader:
    return HuggingFaceDataLoader(metadata, params)


register_loader("huggingface", _factory)
