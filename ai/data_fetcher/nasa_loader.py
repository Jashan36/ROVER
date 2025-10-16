"""
NASA data loader capable of querying multiple APIs (Mars rover images, open
data portal, PDS).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from .auth_manager import AuthManager
from .base_loader import BaseDataLoader, DatasetMetadata, DownloadResult
from .universal_loader import register_loader

logger = logging.getLogger(__name__)


class NASADataLoader(BaseDataLoader):
    """
    Loader for NASA APIs.
    """

    BASE_URL = "https://api.nasa.gov"

    def __init__(self, metadata: DatasetMetadata, params: Dict[str, Any]) -> None:
        self.params = params
        super().__init__(
            metadata,
            cache_dir=params.get("cache_dir"),
            download_dir=params.get("download_dir"),
            force=params.get("force", False),
        )
        self.api_key = AuthManager.setup_credentials("nasa", **params).get(
            "api_key",
            params.get("api_key", "DEMO_KEY"),
        )

    def download(self) -> DownloadResult:
        dataset_id = self.metadata.dataset_id or ""
        if dataset_id.startswith("curiosity/"):
            photos = self._download_mars_rover(dataset_id)
        else:
            photos = self._download_open_data(dataset_id)

        paths = tuple(Path(photo) for photo in photos)
        return DownloadResult(paths=paths, metadata={"count": len(paths)})

    def _download_mars_rover(self, dataset_id: str) -> List[str]:
        rover, endpoint = dataset_id.split("/", 1)
        sol = self.params.get("sol", 1000)
        camera = self.params.get("camera")
        params = {"sol": sol, "api_key": self.api_key}
        if camera:
            params["camera"] = camera

        url = f"{self.BASE_URL}/mars-photos/api/v1/{endpoint}"
        logger.info("Requesting NASA Mars rover imagery: %s", url)
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()

        data = response.json()
        photos = data.get("photos", [])
        download_paths = []
        for photo in photos:
            img_url = photo.get("img_src")
            if not img_url:
                continue
            target_path = self.resolve_cache_path(f"{photo['id']}.jpg")
            if target_path.exists() and not self.force:
                download_paths.append(str(target_path))
                continue
            img_response = requests.get(img_url, timeout=60)
            img_response.raise_for_status()
            target_path.write_bytes(img_response.content)
            download_paths.append(str(target_path))
        return download_paths

    def _download_open_data(self, dataset_id: str) -> List[str]:
        endpoint = self.params.get("endpoint") or dataset_id
        url = endpoint if endpoint.startswith("http") else f"{self.BASE_URL}/{endpoint}"
        logger.info("Downloading NASA open dataset: %s", url)
        response = requests.get(url, timeout=60, params={"api_key": self.api_key})
        response.raise_for_status()
        target_path = self.resolve_cache_path(f"{self.metadata.name or 'nasa_data'}.json")
        target_path.write_text(response.text)
        return [str(target_path)]

    def load(self, **kwargs: Any) -> Any:
        return [path for path in self.resolve_cache_path().glob("*")]

    def preprocess(self, **kwargs: Any) -> Any:
        return None

    def create_splits(self, **kwargs: Any) -> Dict[str, Any]:
        return {}

    def get_statistics(self) -> Dict[str, Any]:
        cache_dir = self.resolve_cache_path()
        files = list(cache_dir.glob("*"))
        return {"num_files": len(files), "paths": [str(f) for f in files[:10]]}


def _factory(metadata: DatasetMetadata, params: Dict[str, Any]) -> BaseDataLoader:
    return NASADataLoader(metadata, params)


register_loader("nasa", _factory)
