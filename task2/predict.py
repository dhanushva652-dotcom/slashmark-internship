from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict cat or dog from a single image.")
    parser.add_argument("--model_path", type=Path, default=Path("models/cat_dog_cnn.keras"))
    parser.add_argument("--image_path", type=Path, required=True)
    parser.add_argument("--img_size", type=int, default=150)
    parser.add_argument("--class_indices", type=Path, default=Path("models/class_indices.json"))
    return parser.parse_args()


def load_and_prepare_image(image_path: Path, img_size: int) -> np.ndarray:
    image = Image.open(image_path).convert("RGB").resize((img_size, img_size))
    array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def main() -> None:
    args = parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    model = keras.models.load_model(args.model_path)

    class_indices = {"cats": 0, "dogs": 1}
    if args.class_indices.exists():
        class_indices = json.loads(args.class_indices.read_text(encoding="utf-8"))

    inv_map = {v: k for k, v in class_indices.items()}

    img = load_and_prepare_image(args.image_path, args.img_size)
    prob = float(model.predict(img, verbose=0)[0][0])
    pred_index = 1 if prob >= 0.5 else 0
    pred_label = inv_map.get(pred_index, str(pred_index))

    confidence = prob if pred_index == 1 else 1 - prob

    print(f"Prediction: {pred_label}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
