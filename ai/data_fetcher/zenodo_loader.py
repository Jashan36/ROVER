"""
Zenodo dataset loader supporting DOI based downloads.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from .base_loader import BaseDataLoader, DatasetMetadata, DownloadResult
from .universal_loader import register_loader

logger = logging.getLogger(__name__)


class ZenodoDataLoader(BaseDataLoader):
    """
    Loader for Zenodo archives accessed via record IDs or DOIs.
    """

    API_ROOT = "https://zenodo.org/api/records"

    def __init__(self, metadata: DatasetMetadata, params: Dict[str, Any]) -> None:
        self.params = params
        super().__init__(
            metadata,
            cache_dir=params.get("cache_dir"),
            download_dir=params.get("download_dir"),
            force=params.get("force", False),
        )

    def download(self) -> DownloadResult:
        record_id = self._resolve_record_id(self.metadata.dataset_id)
        logger.info("Fetching Zenodo record %s", record_id)
        response = requests.get(f"{self.API_ROOT}/{record_id}", timeout=60)
        response.raise_for_status()
        metadata = response.json()
        files = metadata.get("files", [])
        paths = []
        for file_entry in files:
            filename = file_entry["key"]
            url = file_entry["links"]["download"]
            target_path = self.resolve_cache_path(filename)
            if target_path.exists() and not self.force:
                paths.append(target_path)
                continue
            logger.info("Downloading %s", url)
            file_resp = requests.get(url, timeout=120)
            file_resp.raise_for_status()
            target_path.write_bytes(file_resp.content)
            paths.append(target_path)
        return DownloadResult(paths=tuple(paths), metadata={"record": record_id})

    def _resolve_record_id(self, identifier: str) -> str:
        if identifier.lower().startswith("doi:"):
            doi = identifier[4:]
            response = requests.get(
                f"https://doi.org/{doi}",
                timeout=60,
                allow_redirects=True,
            )
            response.raise_for_status()
            return response.url.rstrip("/").split("/")[-1]
        if identifier.isdigit():
            return identifier
        if identifier.startswith("https://"):
            return identifier.rstrip("/").split("/")[-1]
        raise ValueError(f"Unsupported Zenodo identifier: {identifier}")

    def load(self, **kwargs: Any) -> Any:
        return self.resolve_cache_path()

    def preprocess(self, **kwargs: Any) -> Any:
        return None

    def create_splits(self, **kwargs: Any) -> Dict[str, Any]:
        return {}

    def get_statistics(self) -> Dict[str, Any]:
        files = list(self.resolve_cache_path().glob("*"))
        return {"files": len(files), "paths": [str(p) for p in files]}


def _factory(metadata: DatasetMetadata, params: Dict[str, Any]) -> BaseDataLoader:
    return ZenodoDataLoader(metadata, params)


register_loader("zenodo", _factory)
