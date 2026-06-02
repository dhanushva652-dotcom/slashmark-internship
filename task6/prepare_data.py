from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Inspect and optionally extract the provided reference dataset.")
    p.add_argument("--dataset-root", type=str, required=True, help="Path to the extracted reference project.")
    p.add_argument("--extract", action="store_true", help="Extract dataset/imgs.zip into dataset/imgs/")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.dataset_root)
    imgs_zip = root / "dataset" / "imgs.zip"
    data_csv = root / "dataset" / "data.csv"

    if not data_csv.exists():
        raise FileNotFoundError(f"Missing {data_csv}")
    if not imgs_zip.exists():
        raise FileNotFoundError(f"Missing {imgs_zip}")

    print(f"Found CSV: {data_csv}")
    print(f"Found image archive: {imgs_zip}")

    if args.extract:
        out_dir = root / "dataset" / "imgs"
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(imgs_zip, "r") as z:
            z.extractall(out_dir)
        print(f"Extracted to: {out_dir}")


if __name__ == "__main__":
    main()
