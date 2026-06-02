from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class DatasetPaths:
    csv_path: Path
    images_zip_path: Path


def find_dataset_paths(dataset_root: str | Path) -> DatasetPaths:
    root = Path(dataset_root)
    csv_path = root / "dataset" / "data.csv"
    images_zip_path = root / "dataset" / "imgs.zip"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV file: {csv_path}")
    if not images_zip_path.exists():
        raise FileNotFoundError(
            f"Missing image archive: {images_zip_path}. "
            "The reference zip includes imgs.zip inside dataset/."
        )
    return DatasetPaths(csv_path=csv_path, images_zip_path=images_zip_path)


class IndoorObstacleDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        images_zip_path: str | Path,
        train: bool = True,
        image_size: tuple[int, int] = (144, 256),
        augment_flip: bool = True,
    ):
        self.csv_path = Path(csv_path)
        self.images_zip_path = Path(images_zip_path)
        self.train = train
        self.image_size = image_size
        self.augment_flip = augment_flip

        self.df = pd.read_csv(self.csv_path, header=None)
        self.df.columns = ["image", "x", "y", "speed", "angle"]

        self._zip = zipfile.ZipFile(self.images_zip_path, "r")

    def __len__(self) -> int:
        return len(self.df)

    def _read_image(self, rel_path: str) -> Image.Image:
        with self._zip.open(rel_path) as f:
            data = f.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return img

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        img = img.resize((self.image_size[1], self.image_size[0]), Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return tensor

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = self._read_image(str(row["image"]))

        speed = float(row["speed"])
        angle = float(row["angle"])

        if self.train and self.augment_flip and np.random.rand() < 0.5:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            angle = -angle

        # Mild brightness/contrast jitter using OpenCV for stability across indoor lighting.
        if self.train and np.random.rand() < 0.35:
            arr = np.array(img)
            arr = cv2.convertScaleAbs(
                arr,
                alpha=1.0 + np.random.uniform(-0.10, 0.10),
                beta=np.random.uniform(-8, 8),
            )
            img = Image.fromarray(arr)

        x = self._to_tensor(img)
        y = torch.tensor([speed, angle], dtype=torch.float32)
        return x, y
