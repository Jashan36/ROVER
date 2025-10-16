AI folder notes

This folder will contain model code for terrain segmentation and traversability analysis.

Planned files:
- models/unet.py — PyTorch UNet implementation
- infer/run_inference.py — script to run segmentation on downloaded images
- train/train.py — training loop and utils
- utils/preprocess.py — preprocessing helpers for NASA images and DEMs

## Data Fetcher

The `ai.data_fetcher` package now includes a universal loader capable of
downloading datasets from Kaggle, HuggingFace, NASA APIs, Google Drive, Zenodo,
S3, Roboflow, and generic URLs.  Configure default sources in
`ai/config/data_sources.yaml` and run:

```bash
python -m ai.data_fetcher.cli download --auto "https://huggingface.co/datasets/hassanjbara/AI4MARS"
python -m ai.data_fetcher.cli list-sources
```

## Classification Workflow

For raw Mars imagery with CSV metadata (no segmentation masks), use the
classification pipeline:

```bash
# Train directly from CSV-labelled imagery (classification)
python -m ai.training.train_classification \
  --data-root data/raw/mars_dataset/splits \
  --metadata-csv data/raw/mars_dataset/mars_rover_dataset.csv \
  --label-column camera_name \
  --num-epochs 30
```

This leverages `MarsClassificationDataset` and the ResNet-based
`MarsClassifier` to predict labels such as camera names directly from imagery.
