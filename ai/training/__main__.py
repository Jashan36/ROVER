"""
Command-line entry point for training tasks.
Supports both classification and segmentation pipelines.
"""

from __future__ import annotations

import argparse
import logging

from .train_classification import train_classification_model
from .train_segmentation import train_model as train_segmentation_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Mars training utilities")
    subparsers = parser.add_subparsers(dest="task", required=True)

    cls_parser = subparsers.add_parser("classification", help="Train classification model")
    cls_parser.add_argument("--data-root", default="data/raw/mars_dataset/splits")
    cls_parser.add_argument("--metadata-csv", default="data/raw/mars_dataset/mars_rover_dataset.csv")
    cls_parser.add_argument("--label-column", default="camera_name")
    cls_parser.add_argument("--num-epochs", type=int, default=30)
    cls_parser.add_argument("--batch-size", type=int, default=32)
    cls_parser.add_argument("--learning-rate", type=float, default=1e-3)
    cls_parser.add_argument("--num-workers", type=int, default=None)
    cls_parser.add_argument("--device", default="auto")

    seg_parser = subparsers.add_parser("segmentation", help="Train segmentation model")
    seg_parser.add_argument("--data-root", default="data/raw/ai4mars")
    seg_parser.add_argument("--num-epochs", type=int, default=100)
    seg_parser.add_argument("--batch-size", type=int, default=16)
    seg_parser.add_argument("--learning-rate", type=float, default=1e-3)
    seg_parser.add_argument("--num-workers", type=int, default=4)
    seg_parser.add_argument("--device", default="auto")

    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.task == "classification":
        train_classification_model(
            data_root=args.data_root,
            metadata_csv=args.metadata_csv,
            label_column=args.label_column,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_workers=args.num_workers,
            device=args.device,
        )
    elif args.task == "segmentation":
        train_segmentation_model(
            data_root=args.data_root,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
