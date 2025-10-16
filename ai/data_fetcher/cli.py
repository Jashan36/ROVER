"""
Command line interface for the universal data fetcher.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Dict

from . import load_config
from .universal_loader import UniversalDataLoader, detect_source

logger = logging.getLogger(__name__)


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def handle_download(args: argparse.Namespace) -> None:
    loader_kwargs = vars(args)
    loader_kwargs.pop("command")
    loader_kwargs.pop("auto")
    source = loader_kwargs.pop("source", None)
    dataset = loader_kwargs.pop("dataset", None)
    if args.auto and dataset:
        source = detect_source(dataset)
    loader = UniversalDataLoader(source=source, dataset_id=dataset, **loader_kwargs)
    result = loader.download()
    print(json.dumps({"paths": [str(p) for p in result.paths], "cached": result.cached}, indent=2))


def handle_list_sources(args: argparse.Namespace) -> None:
    config = load_config()
    for source, details in config.get("sources", {}).items():
        print(f"{source}:")
        for dataset in details.get("datasets", []):
            print(f"  - {dataset.get('name', dataset.get('id'))}")


def handle_search(args: argparse.Namespace) -> None:
    config = load_config()
    query = args.query.lower()
    matches = []
    for source, details in config.get("sources", {}).items():
        for dataset in details.get("datasets", []):
            if query in json.dumps(dataset).lower():
                matches.append({"source": source, **dataset})
    print(json.dumps(matches, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Universal data fetcher CLI")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download dataset")
    download.add_argument("--source", help="Explicit source type")
    download.add_argument("--dataset", help="Dataset identifier or URL")
    download.add_argument("--auto", action="store_true", help="Auto detect source from identifier")
    download.add_argument("--force", action="store_true", help="Force re-download")
    download.add_argument("--cache-dir", help="Override cache directory")
    download.add_argument("--download-dir", help="Override download directory")
    download.set_defaults(handler=handle_download)

    list_sources = subparsers.add_parser("list-sources", help="List configured sources/datasets")
    list_sources.set_defaults(handler=handle_list_sources)

    search = subparsers.add_parser("search", help="Search configured datasets")
    search.add_argument("query", help="Search term")
    search.set_defaults(handler=handle_search)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    try:
        args.handler(args)
    except Exception as exc:  # pragma: no cover
        logger.exception("Command failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
