from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

from src.data import make_generators
from src.visualize import plot_confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained cat vs dog model.")
    parser.add_argument("--data_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model_path", type=Path, default=Path("models/cat_dog_cnn.keras"))
    parser.add_argument("--img_size", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, val_gen = make_generators(args.data_dir, img_size=args.img_size, batch_size=args.batch_size)
    model = keras.models.load_model(args.model_path)

    val_gen.reset()
    probs = model.predict(val_gen, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)
    y_true = val_gen.classes

    inv_map = {v: k for k, v in val_gen.class_indices.items()}
    class_names = [inv_map[i] for i in sorted(inv_map)]

    print(classification_report(y_true, preds, target_names=class_names))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, preds))

    plot_confusion_matrix(y_true, preds, class_names, Path("outputs/confusion_matrix.png"))


if __name__ == "__main__":
    main()
