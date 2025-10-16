"""Authentication helper utilities for the universal data-loader subsystem."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class AuthManager:
    """
    Central place where we configure credentials/API keys for the various data
    providers.  Authentication data can be provided through environment
    variables, configuration files, or explicit setter methods.
    """

    ENV_MAPPING: Dict[str, Dict[str, str]] = {
        "kaggle": {"username": "KAGGLE_USERNAME", "key": "KAGGLE_KEY"},
        "huggingface": {"token": "HF_TOKEN"},
        "nasa": {"api_key": "NASA_API_KEY"},
        "roboflow": {"api_key": "ROBOFLOW_API_KEY"},
        "s3": {"access_key": "AWS_ACCESS_KEY_ID", "secret_key": "AWS_SECRET_ACCESS_KEY"},
    }

    @classmethod
    def setup_credentials(cls, api_type: str, **kwargs: str) -> Dict[str, str]:
        """
        Return the credentials dictionary for ``api_type``.  The method will
        look for values in ``kwargs`` first, then fall back to environment
        variables, and finally attempt to load provider-specific configuration
        files (e.g. ``~/.kaggle/kaggle.json``).
        """

        credentials = {}
        env_mapping = cls.ENV_MAPPING.get(api_type, {})

        for key, env_var in env_mapping.items():
            value = kwargs.get(key) or os.getenv(env_var)
            if value:
                credentials[key] = value

        if api_type == "kaggle" and not credentials:
            kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
            if kaggle_json.exists():
                try:
                    credentials.update(json.loads(kaggle_json.read_text()))
                except json.JSONDecodeError as exc:
                    logger.warning("Invalid kaggle.json file: %s", exc)

        if not credentials:
            logger.debug("No credentials found for %s", api_type)

        return credentials

    @classmethod
    def require(cls, api_type: str, **kwargs: str) -> Dict[str, str]:
        """
        Ensure credentials are available for ``api_type``.  Raises ``RuntimeError``
        if the provider requires authentication but no credentials are found.
        """

        credentials = cls.setup_credentials(api_type, **kwargs)
        if not credentials and api_type in cls.ENV_MAPPING:
            raise RuntimeError(
                f"Credentials required for {api_type}. "
                f"Set environment variables: {', '.join(cls.ENV_MAPPING[api_type].values())}"
            )

        return credentials
