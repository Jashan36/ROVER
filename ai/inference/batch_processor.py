"""Batch processing script stub for running segmentation on folders of images."""
from pathlib import Path


def process_folder(folder, model):
    folder = Path(folder)
    for p in folder.glob('*.jpg'):
        print('Would process', p)
