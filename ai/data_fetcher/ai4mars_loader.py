"""
Helpers to download / interact with AI4Mars dataset.
Official site: https://ai4mars.org
This module provides download helpers and local indexing utilities.
"""
from pathlib import Path
import requests


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def download_example(dest):
    dest = Path(dest)
    ensure_dir(dest)
    # AI4Mars may provide direct download links; fill in as needed
    return dest
