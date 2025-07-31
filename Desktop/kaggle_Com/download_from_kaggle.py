import zipfile
from pathlib import Path

zip_path = Path("data/raw/house-prices-advanced-regression-techniques.zip")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("data/raw")