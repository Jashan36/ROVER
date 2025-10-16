"""
Amazon S3 loader supporting public and private buckets.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .auth_manager import AuthManager
from .base_loader import BaseDataLoader, DatasetMetadata, DownloadResult
from .universal_loader import register_loader

logger = logging.getLogger(__name__)


class S3DataLoader(BaseDataLoader):
    """
    Loader for Amazon S3 buckets.
    """

    def __init__(self, metadata: DatasetMetadata, params: Dict[str, Any]) -> None:
        self.params = params
        super().__init__(
            metadata,
            cache_dir=params.get("cache_dir"),
            download_dir=params.get("download_dir"),
            force=params.get("force", False),
        )

        import boto3  # type: ignore
        from botocore.config import Config  # type: ignore

        credentials = AuthManager.setup_credentials("s3", **params)
        session_kwargs = {}
        if credentials:
            session_kwargs.update(
                aws_access_key_id=credentials.get("access_key"),
                aws_secret_access_key=credentials.get("secret_key"),
                aws_session_token=credentials.get("session_token"),
            )
        region = params.get("region_name")
        config = Config(signature_version=params.get("signature_version", "s3v4"))
        self.client = boto3.client("s3", region_name=region, config=config, **session_kwargs)

    def download(self) -> DownloadResult:
        bucket, key = self._parse_s3_uri(self.metadata.dataset_id)
        prefix = self.params.get("prefix")
        target_dir = self.resolve_cache_path()

        import boto3  # type: ignore

        logger.info("Syncing from S3 bucket=%s key=%s", bucket, key or prefix)
        paginator = self.client.get_paginator("list_objects_v2")
        operation = paginator.paginate(Bucket=bucket, Prefix=key or prefix)

        paths = []
        for page in operation:
            for obj in page.get("Contents", []):
                s3_key = obj["Key"]
                relative_path = s3_key[len(key) :] if key and s3_key.startswith(key) else s3_key
                destination = target_dir / relative_path
                if destination.exists() and not self.force:
                    paths.append(destination)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                self.client.download_file(bucket, s3_key, str(destination))
                paths.append(destination)

        return DownloadResult(paths=tuple(paths), metadata={"bucket": bucket})

    def _parse_s3_uri(self, uri: str) -> tuple[str, Optional[str]]:
        if uri.startswith("s3://"):
            remainder = uri[5:]
            bucket, _, key = remainder.partition("/")
            return bucket, key or None
        if "/" in uri:
            bucket, key = uri.split("/", 1)
            return bucket, key
        return uri, None

    def load(self, **kwargs: Any) -> Any:
        return self.resolve_cache_path()

    def preprocess(self, **kwargs: Any) -> Any:
        return None

    def create_splits(self, **kwargs: Any) -> Dict[str, Any]:
        return {}

    def get_statistics(self) -> Dict[str, Any]:
        files = list(self.resolve_cache_path().glob("**/*"))
        size_bytes = sum(f.stat().st_size for f in files if f.is_file())
        return {"files": len(files), "size_bytes": size_bytes}


def _factory(metadata: DatasetMetadata, params: Dict[str, Any]) -> BaseDataLoader:
    return S3DataLoader(metadata, params)


register_loader("s3", _factory)
