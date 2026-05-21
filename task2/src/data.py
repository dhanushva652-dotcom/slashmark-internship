from pathlib import Path

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def make_generators(
    data_dir: Path,
    img_size: int = 150,
    batch_size: int = 32,
    validation_split: float = 0.2,
    seed: int = 42,
):

    train_dir = data_dir / "train"
    validation_dir = data_dir / "validation"

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True,
        seed=seed,
    )

    validation_generator = val_datagen.flow_from_directory(
        validation_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )

    return train_generator, validation_generator
