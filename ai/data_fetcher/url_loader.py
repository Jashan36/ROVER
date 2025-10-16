"""
Generic HTTP/FTP URL loader.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from tqdm import tqdm

from .base_loader import BaseDataLoader, DatasetMetadata, DownloadResult
from .universal_loader import register_loader

logger = logging.getLogger(__name__)


class URLDataLoader(BaseDataLoader):
    """
    Loader that downloads datasets from direct URLs.
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
        url = self.params.get("url") or self.metadata.dataset_id
        filename = self.params.get("filename") or url.split("/")[-1]
        target_path = self.resolve_cache_path(filename)

        if target_path.exists() and not self.force:
            logger.info("URL already downloaded: %s", target_path)
            return DownloadResult(paths=(target_path,), cached=True)

        logger.info("Downloading %s", url)
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            with open(target_path, "wb") as fh, tqdm(total=total, unit="B", unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    pbar.update(len(chunk))

        return DownloadResult(paths=(target_path,), metadata={"url": url})

    def load(self, **kwargs: Any) -> Any:
        return self.resolve_cache_path()

    def preprocess(self, **kwargs: Any) -> Any:
        return None

    def create_splits(self, **kwargs: Any) -> Dict[str, Any]:
        return {}

    def get_statistics(self) -> Dict[str, Any]:
        files = list(self.resolve_cache_path().glob("*"))
        return {"files": [str(p) for p in files]}


def _factory(metadata: DatasetMetadata, params: Dict[str, Any]) -> BaseDataLoader:
    return URLDataLoader(metadata, params)


register_loader("url", _factory)
