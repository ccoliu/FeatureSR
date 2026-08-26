"""
download_datasets.py
資料集下載輔助腳本

提供 EuroSAT 自動下載（其他資料集需手動下載）。
"""
import os
import sys
import argparse
import zipfile
import urllib.request
from pathlib import Path
from tqdm import tqdm
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_eurosat(data_root: Path):
    """下載 EuroSAT 資料集"""
    target_dir = data_root / "EuroSAT"
    if (target_dir / "2750").exists():
        print(f"EuroSAT already exists at {target_dir}")
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    url = "https://madm.dfki.de/files/sentinel/EuroSAT.zip"
    zip_path = target_dir / "EuroSAT.zip"

    print(f"Downloading EuroSAT from {url}...")
    try:
        download_url(url, str(zip_path))
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(target_dir)
        zip_path.unlink()
        print(f"EuroSAT downloaded to {target_dir}")
    except Exception as e:
        print(f"Error downloading EuroSAT: {e}")
        print("Please download manually from: https://madm.dfki.de/files/sentinel/EuroSAT.zip")


def show_manual_download_instructions():
    """顯示需手動下載的資料集說明"""
    instructions = """
=====================================================================
Manual Download Instructions for CDFSL Benchmark Datasets
=====================================================================

1. CropDiseases (Plant Village)
   URL: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
   Structure: data/CropDiseases/dataset/<class_name>/*.jpg

2. ISIC2018 (Skin Lesion)
   URL: https://challenge2018.isic-archive.com/
   - Download "Training Data" (ISIC_2018_Task3_Training_Input.zip)
   - Download "Training Groundtruth" (ISIC2018_Task3_Training_GroundTruth.zip)
   Structure:
     data/ISIC2018/ISIC2018_Task3_Training_Input/*.jpg
     data/ISIC2018/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv

3. ChestX-ray8
   URL: https://nihcc.app.box.com/v/ChestXray-NIHCC
   - Download all images_XXX.tar.gz files and Data_Entry_2017.csv
   Structure:
     data/ChestX/images/*.png
     data/ChestX/Data_Entry_2017.csv

4. EuroSAT (auto-downloadable)
   Run: python download_datasets.py --dataset eurosat --data_root ./data

=====================================================================
"""
    print(instructions)


def main():
    parser = argparse.ArgumentParser(description="Download CDFSL datasets")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "eurosat", "crop_disease", "isic", "chestx"])
    parser.add_argument("--data_root", type=str, default="./data")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("eurosat", "all"):
        download_eurosat(data_root)

    if args.dataset in ("crop_disease", "isic", "chestx", "all"):
        show_manual_download_instructions()


if __name__ == "__main__":
    main()
