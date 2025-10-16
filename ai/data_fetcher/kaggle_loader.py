"""
Kaggle dataset loader implementation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .auth_manager import AuthManager
from .base_loader import BaseDataLoader, DatasetMetadata, DownloadResult
from .universal_loader import register_loader

logger = logging.getLogger(__name__)


class KaggleDataLoader(BaseDataLoader):
    """
    Loader for Kaggle datasets and competitions.
    """

    def __init__(self, metadata: DatasetMetadata, params: Dict[str, Any]) -> None:
        self.params = params
        super().__init__(
            metadata,
            cache_dir=params.get("cache_dir"),
            download_dir=params.get("download_dir"),
            force=params.get("force", False),
        )
        self.credentials = AuthManager.setup_credentials("kaggle", **params)

    # ------------------------------------------------------------------
    def download(self) -> DownloadResult:
        import kagglehub  # type: ignore

        dataset_ref = self.metadata.dataset_id
        target_dir = self.resolve_cache_path()

        if self.cache_exists() and not self.force:
            logger.info("Kaggle dataset cached; reusing %s", target_dir)
            return DownloadResult(paths=(target_dir,), cached=True)

        logger.info("Downloading Kaggle dataset %s", dataset_ref)
        if dataset_ref.startswith("competition:") or self.params.get("competition"):
            competition_ref = dataset_ref.split(":", 1)[-1]
            download_path = kagglehub.competition_download(competition_ref, path=str(target_dir))
        else:
            download_path = kagglehub.dataset_download(dataset_ref, path=str(target_dir))

        if isinstance(download_path, str):
            download_path = Path(download_path)

        return DownloadResult(paths=(Path(download_path),), metadata={"dataset": dataset_ref})

    def load(self, **kwargs: Any) -> Any:
        logger.info("Kaggle loader returning download directory for manual loading")
        return self.resolve_cache_path()

    def preprocess(self, **kwargs: Any) -> Any:
        logger.debug("No preprocessing implemented for Kaggle loader")
        return None

    def create_splits(self, **kwargs: Any) -> Dict[str, Any]:
        return {}

    def get_statistics(self) -> Dict[str, Any]:
        dataset_path = self.resolve_cache_path()
        size_bytes = sum(p.stat().st_size for p in dataset_path.rglob("*") if p.is_file())
        return {"path": str(dataset_path), "size_bytes": size_bytes}


def _factory(metadata: DatasetMetadata, params: Dict[str, Any]) -> BaseDataLoader:
    return KaggleDataLoader(metadata, params)


register_loader("kaggle", _factory)
