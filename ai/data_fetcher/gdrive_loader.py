"""
Google Drive loader using gdown.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from .base_loader import BaseDataLoader, DatasetMetadata, DownloadResult
from .dependency_manager import DependencyManager
from .universal_loader import register_loader

logger = logging.getLogger(__name__)


class GoogleDriveLoader(BaseDataLoader):
    """
    Loader for files hosted on Google Drive.
    """

    def __init__(self, metadata: DatasetMetadata, params: Dict[str, Any]) -> None:
        self.params = params
        super().__init__(
            metadata,
            cache_dir=params.get("cache_dir"),
            download_dir=params.get("download_dir"),
            force=params.get("force", False),
        )

    def download(self) -> DownloadResult:
        DependencyManager.check("gdrive")
        import gdown  # type: ignore

        file_id = self.params.get("file_id") or self.metadata.dataset_id
        target_name = self.params.get("filename") or f"{file_id}.download"
        target_path = self.resolve_cache_path(target_name)

        if target_path.exists() and not self.force:
            return DownloadResult(paths=(target_path,), cached=True)

        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(target_path), quiet=False)
        return DownloadResult(paths=(target_path,), metadata={"file_id": file_id})

    def load(self, **kwargs: Any) -> Any:
        return self.resolve_cache_path()

    def preprocess(self, **kwargs: Any) -> Any:
        return None

    def create_splits(self, **kwargs: Any) -> Dict[str, Any]:
        return {}

    def get_statistics(self) -> Dict[str, Any]:
        target_path = self.resolve_cache_path()
        files = list(target_path.glob("*"))
        return {"num_files": len(files), "paths": [str(p) for p in files]}


def _factory(metadata: DatasetMetadata, params: Dict[str, Any]) -> BaseDataLoader:
    return GoogleDriveLoader(metadata, params)


register_loader("gdrive", _factory)
