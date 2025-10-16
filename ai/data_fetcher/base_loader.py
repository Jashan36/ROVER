"""
Base interfaces for universal data loader subsystem.

This module defines the common abstractions shared across the different
data-source specific loaders (Kaggle, HuggingFace, NASA, etc.).  All loader
implementations must inherit from :class:`BaseDataLoader` so the
``UniversalDataLoader`` can drive them via a consistent, highly extensible API.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DatasetMetadata:
    """
    Canonical description for datasets handled by the loader system.

    Attributes:
        source: Canonical source identifier (e.g. ``kaggle``).
        dataset_id: Source specific identifier (e.g. ``user/dataset``).
        name: Optional human friendly name.
        version: Optional version string.
        description: Optional dataset description.
        tags: Optional set of descriptive tags.
    """

    source: str
    dataset_id: str
    name: Optional[str] = None
    version: Optional[str] = None
    description: Optional[str] = None
    tags: Tuple[str, ...] = field(default_factory=tuple)


@dataclass
class DownloadResult:
    """
    Result container returned by :meth:`BaseDataLoader.download`.

    Attributes:
        paths: Paths to downloaded files or directories.
        metadata: Optional metadata describing download artefacts.
        cached: True if data was already available locally.
    """

    paths: Tuple[Path, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)
    cached: bool = False


class LoaderHook:
    """
    Lightweight hook representation for loader lifecycle events.

    Hooks can be registered by plugins or user code to observe or mutate the
    loader workflow.  Each hook receives the loader instance along with
    keyword arguments relevant to the lifecycle stage.
    """

    def __init__(self) -> None:
        self._callbacks: List[Callable[..., None]] = []

    def register(self, callback: Callable[..., None]) -> None:
        """Register a new callback."""
        self._callbacks.append(callback)

    def fire(self, **kwargs: Any) -> None:
        """Invoke all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(**kwargs)
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Loader hook callback failed", extra={"callback": callback})


class BaseDataLoader(ABC):
    """
    Abstract base class for all dataset loaders.

    Implementations should inherit from this class and implement the abstract
    methods to provide source-specific behaviour.  Concrete loaders are
    expected to download raw data, optionally preprocess it, and expose uniform
    metadata/statistics.
    """

    # Hook registry (per-instance)
    def __init__(
        self,
        metadata: DatasetMetadata,
        cache_dir: Optional[Path] = None,
        download_dir: Optional[Path] = None,
        force: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            metadata: Dataset metadata descriptor.
            cache_dir: Directory used for cached artefacts (defaults to ~/.cache).
            download_dir: Directory where downloads should be stored.
            force: If True, force re-download even when cached data exists.
            **kwargs: Additional parameters passed to subclass initialisers.
        """

        self.metadata = metadata
        self.force = force
        self.cache_dir = Path(cache_dir or Path.home() / ".cache" / "space_rover")
        self.download_dir = Path(download_dir or self.cache_dir / "downloads" / metadata.source)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.extra_args = kwargs

        # Loader pipeline hooks
        self.on_before_download = LoaderHook()
        self.on_after_download = LoaderHook()
        self.on_before_preprocess = LoaderHook()
        self.on_after_preprocess = LoaderHook()

        logger.debug(
            "Initialised BaseDataLoader",
            extra={
                "metadata": self.metadata,
                "cache_dir": str(self.cache_dir),
                "download_dir": str(self.download_dir),
                "force": self.force,
            },
        )

    # ------------------------------------------------------------------ #
    # Abstract interface
    # ------------------------------------------------------------------ #
    @abstractmethod
    def download(self) -> DownloadResult:
        """
        Download the dataset artefacts for the loader.
        """

    @abstractmethod
    def load(self, **kwargs: Any) -> Any:
        """
        Load the dataset into memory or return dataset iterables ready for
        consumption (e.g. torch DataLoaders, HuggingFace Dataset objects).
        """

    @abstractmethod
    def preprocess(self, **kwargs: Any) -> Any:
        """
        Perform optional preprocessing operations.
        """

    @abstractmethod
    def create_splits(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Create train/validation/test splits if supported by the dataset.
        """

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Return dataset statistics (size, number of classes, etc.).
        """

    # ------------------------------------------------------------------ #
    # Convenience helpers
    # ------------------------------------------------------------------ #
    def cache_exists(self) -> bool:
        """
        Returns True if cached data exists for the dataset.
        """
        cache_path = self.download_dir / self.metadata.dataset_id.replace("/", "_")
        exists = cache_path.exists()
        logger.debug(
            "Cache check for dataset",
            extra={
                "dataset": self.metadata.dataset_id,
                "cache_path": str(cache_path),
                "exists": exists,
            },
        )
        return exists

    def resolve_cache_path(self, *parts: str) -> Path:
        """
        Build a path within the dataset-specific cache directory.
        """
        base = self.download_dir / self.metadata.dataset_id.replace("/", "_")
        base.mkdir(parents=True, exist_ok=True)
        resolved = base.joinpath(*parts)
        logger.debug("Resolved cache path", extra={"path": str(resolved)})
        return resolved

    def register_hook(self, stage: str, callback: Callable[..., None]) -> None:
        """
        Register a hook callback for a lifecycle stage.
        """
        hook_map = {
            "before_download": self.on_before_download,
            "after_download": self.on_after_download,
            "before_preprocess": self.on_before_preprocess,
            "after_preprocess": self.on_after_preprocess,
        }

        if stage not in hook_map:
            raise ValueError(f"Unknown hook stage '{stage}'")

        hook_map[stage].register(callback)

    def run_pipeline(self) -> Dict[str, Any]:
        """
        Execute the default loader pipeline workflow.
        """
        logger.info("Starting loader pipeline", extra={"dataset": self.metadata.dataset_id})

        if self.cache_exists() and not self.force:
            logger.info("Cache available, skipping download", extra={"dataset": self.metadata.dataset_id})
            download_result = DownloadResult(paths=(self.resolve_cache_path(),), cached=True)
        else:
            self.on_before_download.fire(loader=self, metadata=self.metadata)
            download_result = self.download()
            self.on_after_download.fire(loader=self, result=download_result)

        self.on_before_preprocess.fire(loader=self, download_result=download_result)
        preprocess_result = self.preprocess()
        self.on_after_preprocess.fire(loader=self, preprocess_result=preprocess_result)

        splits = self.create_splits()
        stats = self.get_statistics()

        return {
            "download": download_result,
            "preprocess": preprocess_result,
            "splits": splits,
            "statistics": stats,
        }


LoaderFactory = Callable[[DatasetMetadata, Dict[str, Any]], BaseDataLoader]
