from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import classification_report
from tensorflow import keras

from src.data import make_generators
from src.model import build_cnn_model
from src.visualize import plot_confusion_matrix, plot_training_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN for Cats vs Dogs")

    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("dataset"),
        help="Dataset directory",
    )

    parser.add_argument(
        "--img_size",
        type=int,
        default=150,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--model_out",
        type=Path,
        default=Path("models/cat_dog_cnn.keras"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_path = args.data_dir / "train"
    val_path = args.data_dir / "validation"

    print("\n========== DATASET DEBUG ==========")
    print("Current Directory:", Path.cwd())
    print("Dataset Path:", args.data_dir.resolve())

    print("Train Exists:", train_path.exists())
    print("Validation Exists:", val_path.exists())

    if train_path.exists():
        print("Train Cats:", len(list((train_path / "cats").glob("*"))))
        print("Train Dogs:", len(list((train_path / "dogs").glob("*"))))

    if val_path.exists():
        print("Validation Cats:", len(list((val_path / "cats").glob("*"))))
        print("Validation Dogs:", len(list((val_path / "dogs").glob("*"))))

    print("===================================\n")

    train_gen, val_gen = make_generators(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        seed=args.seed,
    )

    if train_gen.samples == 0 or val_gen.samples == 0:
        raise ValueError(
            "No images found in dataset folders."
        )

    model = build_cnn_model((args.img_size, args.img_size, 3))

    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(args.model_out),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),

        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    args.model_out.parent.mkdir(parents=True, exist_ok=True)

    model.save(args.model_out)

    class_indices_path = args.model_out.parent / "class_indices.json"

    with class_indices_path.open("w") as f:
        json.dump(train_gen.class_indices, f, indent=2)

    plot_training_history(
        history,
        Path("outputs/training_curves.png"),
    )

    val_gen.reset()

    probs = model.predict(val_gen).ravel()

    preds = (probs >= 0.5).astype(int)

    y_true = val_gen.classes

    inv_map = {
        v: k for k, v in train_gen.class_indices.items()
    }

    class_names = [
        inv_map[i] for i in sorted(inv_map)
    ]

    plot_confusion_matrix(
        y_true,
        preds,
        class_names,
        Path("outputs/confusion_matrix.png"),
    )

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_true,
            preds,
            target_names=class_names,
        )
    )

    metrics = model.evaluate(val_gen)

    print(f"\nValidation Loss: {metrics[0]:.4f}")
    print(f"Validation Accuracy: {metrics[1]:.4f}")

    print(f"\nModel Saved To: {args.model_out.resolve()}")


if __name__ == "__main__":
    main()