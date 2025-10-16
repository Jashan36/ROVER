"""
Roboflow dataset loader.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from .auth_manager import AuthManager
from .base_loader import BaseDataLoader, DatasetMetadata, DownloadResult
from .dependency_manager import DependencyManager
from .universal_loader import register_loader

logger = logging.getLogger(__name__)


class RoboflowDataLoader(BaseDataLoader):
    """
    Loader for Roboflow computer vision datasets.
    """

    def __init__(self, metadata: DatasetMetadata, params: Dict[str, Any]) -> None:
        self.params = params
        super().__init__(
            metadata,
            cache_dir=params.get("cache_dir"),
            download_dir=params.get("download_dir"),
            force=params.get("force", False),
        )

        credentials = AuthManager.require("roboflow", **params)
        self.api_key = credentials["api_key"]

    def download(self) -> DownloadResult:
        DependencyManager.check("roboflow")
        from roboflow import Roboflow  # type: ignore

        dataset_id = self.metadata.dataset_id
        workspace, dataset_name = dataset_id.split("/")
        version = self.params.get("version")
        format_ = self.params.get("format", "yolov5")

        rf = Roboflow(api_key=self.api_key)
        dataset = rf.workspace(workspace).project(dataset_name)
        if version:
            dataset = dataset.version(version)
        download_path = dataset.download(format_)
        path = Path(download_path)
        target_dir = self.resolve_cache_path()
        if path.is_dir():
            for item in path.iterdir():
                target = target_dir / item.name
                if item.is_dir():
                    if target.exists():
                        continue
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.write_bytes(item.read_bytes())
        return DownloadResult(paths=(target_dir,), metadata={"dataset": dataset_id})

    def load(self, **kwargs: Any) -> Any:
        return self.resolve_cache_path()

    def preprocess(self, **kwargs: Any) -> Any:
        return None

    def create_splits(self, **kwargs: Any) -> Dict[str, Any]:
        return {}

    def get_statistics(self) -> Dict[str, Any]:
        files = list(self.resolve_cache_path().glob("**/*"))
        return {"num_files": len(files)}


def _factory(metadata: DatasetMetadata, params: Dict[str, Any]) -> BaseDataLoader:
    return RoboflowDataLoader(metadata, params)


register_loader("roboflow", _factory)
