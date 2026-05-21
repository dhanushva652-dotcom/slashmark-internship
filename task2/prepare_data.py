from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def split_dataset(raw_dir: Path, output_dir: Path, val_ratio: float = 0.2, seed: int = 42) -> None:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    images = [p for p in raw_dir.iterdir() if p.is_file() and _is_image(p)]
    if not images:
        raise ValueError(
            f"No images found in {raw_dir}. Put Kaggle Dogs vs Cats images there first."
        )

    cats = [p for p in images if p.name.lower().startswith("cat")]
    dogs = [p for p in images if p.name.lower().startswith("dog")]

    if not cats or not dogs:
        raise ValueError(
            "Could not find both cat and dog images. Filenames should start with 'cat' or 'dog'."
        )

    rng = random.Random(seed)
    rng.shuffle(cats)
    rng.shuffle(dogs)

    def split_list(items: list[Path]) -> tuple[list[Path], list[Path]]:
        val_count = max(1, int(len(items) * val_ratio))
        return items[val_count:], items[:val_count]

    train_cats, val_cats = split_list(cats)
    train_dogs, val_dogs = split_list(dogs)

    layout = {
        "train/cats": train_cats,
        "train/dogs": train_dogs,
        "val/cats": val_cats,
        "val/dogs": val_dogs,
    }

    if output_dir.exists():
        shutil.rmtree(output_dir)
    for folder in layout:
        (output_dir / folder).mkdir(parents=True, exist_ok=True)

    def copy_files(files: Iterable[Path], target_dir: Path) -> None:
        for src in files:
            shutil.copy2(src, target_dir / src.name)

    copy_files(train_cats, output_dir / "train" / "cats")
    copy_files(train_dogs, output_dir / "train" / "dogs")
    copy_files(val_cats, output_dir / "val" / "cats")
    copy_files(val_dogs, output_dir / "val" / "dogs")

    print("Dataset prepared successfully.")
    print(f"Train cats: {len(train_cats)}")
    print(f"Train dogs: {len(train_dogs)}")
    print(f"Val cats:   {len(val_cats)}")
    print(f"Val dogs:   {len(val_dogs)}")
    print(f"Output folder: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split raw cat/dog images into train and validation folders.")
    parser.add_argument("--raw_dir", type=Path, default=Path("data/raw"), help="Folder with raw cat/dog images")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed"),
        help="Destination folder for train/val split",
    )
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split_dataset(args.raw_dir, args.output_dir, args.val_ratio, args.seed)
