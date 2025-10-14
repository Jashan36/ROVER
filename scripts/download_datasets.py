"""
Download scripts for real planetary datasets.
This script doesn't download large files by default; it provides commands and helper functions.

Datasets referenced:
- AI4Mars: https://ai4mars.org
- HiRISE / USGS DEM: https://astrogeology.usgs.gov/search
- Perseverance raw images (NASA/JPL): https://mars.nasa.gov/mars2020/multimedia/raw-images/

This script is a small helper to download single images or list of URLs. For large dataset downloads use the official dataset pages.
"""

import os
import requests
from urllib.parse import urljoin

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

def download_file(url, out_path):
    out_path = os.path.abspath(out_path)
    print(f"Downloading {url} -> {out_path}")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(out_path, 'wb') as f:
        for chunk in r.iter_content(1024*1024):
            f.write(chunk)
    print('Done')

if __name__ == '__main__':
    print('This helper can be used to download individual files.')
    print('Examples:')
    print('python download_datasets.py https://mars.nasa.gov/system/resources/detail_files/000/080/123/abcd.jpg data/abcd.jpg')
