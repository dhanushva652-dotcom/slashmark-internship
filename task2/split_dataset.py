import os
import shutil
import random

SOURCE_DIR = "data"

BASE_DIR = "dataset"

train_cats = os.path.join(BASE_DIR, "train/cats")
train_dogs = os.path.join(BASE_DIR, "train/dogs")
val_cats = os.path.join(BASE_DIR, "validation/cats")
val_dogs = os.path.join(BASE_DIR, "validation/dogs")

for folder in [train_cats, train_dogs, val_cats, val_dogs]:
    os.makedirs(folder, exist_ok=True)

cats = os.listdir(os.path.join(SOURCE_DIR, "cats"))
dogs = os.listdir(os.path.join(SOURCE_DIR, "dogs"))

random.shuffle(cats)
random.shuffle(dogs)

split_ratio = 0.8

cat_split = int(len(cats) * split_ratio)
dog_split = int(len(dogs) * split_ratio)

train_cat_files = cats[:cat_split]
val_cat_files = cats[cat_split:]

train_dog_files = dogs[:dog_split]
val_dog_files = dogs[dog_split:]

def copy_files(files, source_folder, target_folder):
    for file in files:
        src = os.path.join(source_folder, file)
        dst = os.path.join(target_folder, file)
        shutil.copy(src, dst)

copy_files(
    train_cat_files,
    os.path.join(SOURCE_DIR, "cats"),
    train_cats
)

copy_files(
    val_cat_files,
    os.path.join(SOURCE_DIR, "cats"),
    val_cats
)

copy_files(
    train_dog_files,
    os.path.join(SOURCE_DIR, "dogs"),
    train_dogs
)

copy_files(
    val_dog_files,
    os.path.join(SOURCE_DIR, "dogs"),
    val_dogs
)

print("Dataset split completed successfully!")