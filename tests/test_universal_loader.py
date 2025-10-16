import types
from pathlib import Path
from typing import Any, Dict

import pytest

from ai.data_fetcher.base_loader import BaseDataLoader, DatasetMetadata, DownloadResult
from ai.data_fetcher.universal_loader import (
    UniversalDataLoader,
    detect_source,
    load_config,
    register_loader,
)


class DummyLoader(BaseDataLoader):
    def __init__(self, metadata: DatasetMetadata, params: Dict[str, Any]) -> None:
        super().__init__(metadata, **params)
        self._downloaded = False

    def download(self) -> DownloadResult:
        self._downloaded = True
        path = self.resolve_cache_path("dummy.txt")
        path.write_text("dummy")
        return DownloadResult(paths=(path,), cached=False)

    def load(self, **kwargs: Any) -> Any:
        return "data"

    def preprocess(self, **kwargs: Any) -> Any:
        return None

    def create_splits(self, **kwargs: Any) -> Dict[str, Any]:
        return {}

    def get_statistics(self) -> Dict[str, Any]:
        return {"downloaded": self._downloaded}


def dummy_factory(metadata: DatasetMetadata, params: Dict[str, Any]) -> BaseDataLoader:
    return DummyLoader(metadata, params)


def test_detect_source_variants():
    assert detect_source("https://kaggle.com/user/dataset") == "kaggle"
    assert detect_source("hassanjbara/AI4MARS") == "kaggle"  # falls back to kaggle-style id
    assert detect_source("https://huggingface.co/datasets/foo/bar") == "huggingface"
    assert detect_source("https://drive.google.com/file/d/123/view") == "gdrive"
    assert detect_source("s3://bucket/path") == "s3"
    assert detect_source("https://zenodo.org/record/4419131") == "zenodo"


def test_universal_loader_registration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    register_loader("dummy", dummy_factory)

    loader = UniversalDataLoader(
        source="dummy",
        dataset_id="example",
        cache_dir=tmp_path,
        download_dir=tmp_path,
    )

    base_loader = loader.get_loader()
    assert isinstance(base_loader, DummyLoader)

    result = loader.download()
    assert result.paths
    assert (tmp_path / "example" / "dummy.txt").exists()


def test_load_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_path = tmp_path / "data_sources.yaml"
    config_path.write_text(
        """
sources:
  dummy:
    datasets:
      - id: "id"
        name: "name"
"""
    )

    config = load_config(config_path)
    assert "dummy" in config["sources"]
