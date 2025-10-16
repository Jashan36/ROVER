"""
Universal data-loader factory supporting multiple external data providers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

import yaml

from .base_loader import BaseDataLoader, DatasetMetadata, LoaderFactory
from .dependency_manager import DependencyManager

logger = logging.getLogger(__name__)

_REGISTRY: Dict[str, LoaderFactory] = {}


def register_loader(source: str, factory: LoaderFactory) -> None:
    """
    Register a new loader factory for ``source``.
    """
    source_key = source.lower()
    if source_key in _REGISTRY:
        logger.warning("Overwriting loader registration for %s", source_key)
    _REGISTRY[source_key] = factory


def get_registered_loader(source: str) -> LoaderFactory:
    try:
        return _REGISTRY[source.lower()]
    except KeyError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"No loader registered for source '{source}'") from exc


def detect_source(identifier: str) -> Optional[str]:
    """
    Infer the data source type from an identifier/URL.
    """
    lowered = identifier.lower()
    if "kaggle.com" in lowered or "/" in identifier and len(identifier.split("/")) == 2:
        return "kaggle"
    if "huggingface.co" in lowered:
        return "huggingface"
    if "drive.google.com" in lowered or "googledrive.com" in lowered:
        return "gdrive"
    if lowered.startswith(("s3://", "arn:aws:s3")):
        return "s3"
    if "roboflow.com" in lowered:
        return "roboflow"
    if lowered.startswith("doi:") or "zenodo.org" in lowered:
        return "zenodo"
    if lowered.startswith("ftp://"):
        return "url"
    if lowered.startswith("http://") or lowered.startswith("https://"):
        if "nasa.gov" in lowered:
            return "nasa"
        return "url"
    return None


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load the universal data-loader configuration.
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "data_sources.yaml"
    if not config_path.exists():
        logger.debug("Configuration file %s not found; returning empty config", config_path)
        return {}
    return yaml.safe_load(config_path.read_text()) or {}


class UniversalDataLoader:
    """
    Factory wrapper that instantiates the appropriate loader implementation for
    the requested source/dataset.
    """

    def __init__(
        self,
        source: Optional[str] = None,
        dataset_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        auto_install_deps: bool = True,
        **kwargs: Any,
    ) -> None:
        if not source and dataset_id:
            source = detect_source(dataset_id)
        if not source:
            raise ValueError("Source could not be determined. Provide `source` explicitly.")

        self.source = source.lower()
        self.dataset_id = dataset_id
        self.extra_args = kwargs
        self.auto_install_deps = auto_install_deps
        self.config = config or load_config()

        logger.debug(
            "UniversalDataLoader initialised",
            extra={"source": self.source, "dataset_id": dataset_id, "extra_args": kwargs},
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_url(cls, url: str, **kwargs: Any) -> "UniversalDataLoader":
        return cls(dataset_id=url, **kwargs)

    @classmethod
    def from_config(cls, name: str, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> "UniversalDataLoader":
        cfg = config or load_config()
        for source, details in cfg.get("sources", {}).items():
            for entry in details.get("datasets", []):
                if entry.get("name") == name:
                    return cls(
                        source=source,
                        dataset_id=entry.get("id") or entry.get("url"),
                        config=cfg,
                        **{**entry, **kwargs},
                    )
        raise ValueError(f"Dataset '{name}' not found in configuration")

    # ------------------------------------------------------------------
    def auto_detect_source(self) -> str:
        detected = detect_source(self.dataset_id or "")
        if not detected:
            raise ValueError(f"Unable to auto-detect source for identifier '{self.dataset_id}'")
        return detected

    def get_loader(self) -> BaseDataLoader:
        """
        Instantiate the registered loader for the requested data source.
        """
        if hasattr(self, "_loader") and self._loader is not None:  # type: ignore[attr-defined]
            return self._loader  # type: ignore[attr-defined]

        DependencyManager.check(self.source, auto_install=self.auto_install_deps)

        metadata = DatasetMetadata(
            source=self.source,
            dataset_id=self.dataset_id or self.extra_args.get("dataset_id", ""),
            name=self.extra_args.get("name"),
            version=self.extra_args.get("version"),
            description=self.extra_args.get("description"),
        )

        loader_factory = get_registered_loader(self.source)
        loader = loader_factory(metadata, self.extra_args)
        self._loader = loader  # type: ignore[attr-defined]
        return loader

    # Convenience passthroughs -------------------------------------------------
    def download(self) -> Any:
        return self.get_loader().download()

    def load(self) -> Any:
        return self.get_loader().load()

    def preprocess(self) -> Any:
        return self.get_loader().preprocess()

    def create_splits(self) -> Any:
        return self.get_loader().create_splits()

    def get_statistics(self) -> Any:
        return self.get_loader().get_statistics()
