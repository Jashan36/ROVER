"""
Utility helpers for managing optional third-party dependencies required by
individual data-source loaders.

The dependency manager performs lazy import checks and optionally installs
packages on demand.  Automatic installation can be disabled by setting the
environment variable ``SPACE_ROVER_AUTO_INSTALL=0``.
"""

from __future__ import annotations

import importlib
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DependencySpec:
    """Description for a dependency requirement."""

    packages: Iterable[str]
    install_command: Optional[List[str]] = None


class DependencyManager:
    """
    Central registry responsible for dependency verification/installation.
    """

    DEFAULT_DEPENDENCIES: Dict[str, DependencySpec] = {
        "kaggle": DependencySpec(packages=["kagglehub", "kaggle"]),
        "huggingface": DependencySpec(packages=["datasets", "huggingface_hub"]),
        "gdrive": DependencySpec(packages=["gdown"]),
        "s3": DependencySpec(packages=["boto3"]),
        "nasa": DependencySpec(packages=["requests"]),
        "roboflow": DependencySpec(packages=["roboflow"]),
        "zenodo": DependencySpec(packages=["requests"]),
        "url": DependencySpec(packages=["requests"]),
    }

    @classmethod
    def check(cls, api_type: str, auto_install: bool = True) -> bool:
        """
        Verify that dependencies for ``api_type`` are available.
        """
        spec = cls.DEFAULT_DEPENDENCIES.get(api_type)
        if not spec:
            logger.debug("No dependencies registered for type %s", api_type)
            return True

        missing = cls._missing_packages(spec.packages)
        if not missing:
            return True

        if not auto_install or os.getenv("SPACE_ROVER_AUTO_INSTALL", "1") == "0":
            logger.warning("Missing dependencies for %s: %s", api_type, ", ".join(missing))
            return False

        logger.info("Installing missing dependencies for %s", api_type)
        cls._install_packages(missing, spec.install_command)

        post_missing = cls._missing_packages(spec.packages)
        if post_missing:
            raise RuntimeError(
                f"Dependencies for {api_type} still missing after installation: {', '.join(post_missing)}"
            )

        return True

    @staticmethod
    def _missing_packages(packages: Iterable[str]) -> List[str]:
        missing = []
        for name in packages:
            try:
                importlib.import_module(name)
            except ModuleNotFoundError:
                missing.append(name)
        return missing

    @staticmethod
    def _install_packages(packages: Iterable[str], install_command: Optional[List[str]]) -> None:
        command = install_command or [sys.executable, "-m", "pip", "install", *packages]
        logger.debug("Executing dependency installation command", extra={"command": command})

        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("Dependency installation failed: %s", result.stderr.strip())
            raise RuntimeError(f"Failed to install dependencies: {packages}")

        logger.info("Dependencies installed successfully: %s", ", ".join(packages))
