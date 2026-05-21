# Dogs vs Cats CNN Classifier

A complete TensorFlow/Keras project that trains a CNN to classify **cat** vs **dog** images.

## What this project includes
- CNN model built with Keras
- Data preparation script for Kaggle-style raw images
- Image augmentation with `ImageDataGenerator`
- Training and validation curves
- Confusion matrix evaluation
- Single-image prediction script

## Dataset
This project is designed for the Kaggle **Dogs vs Cats** dataset.

Expected raw input format:
```text
data/raw/
  cat.0.jpg
  cat.1.jpg
  dog.0.jpg
  dog.1.jpg
  ...
```

## Recommended environment
Use **Python 3.10 or 3.11** with TensorFlow 2.x.

## Install dependencies
```bash
pip install -r requirements.txt
```

## Step 1: Prepare the data
This splits the raw Kaggle images into train/validation folders.

```bash
python prepare_data.py --raw_dir data/raw --output_dir data/processed --val_ratio 0.2
```

It creates:
```text
data/processed/
  train/cats
  train/dogs
  val/cats
  val/dogs
```

## Step 2: Train the model
```bash
python train.py --data_dir data/processed --epochs 15 --batch_size 32 --img_size 150
```

Outputs are saved in:
```text
models/cat_dog_cnn.keras
models/class_indices.json
outputs/training_curves.png
outputs/confusion_matrix.png
```

## Step 3: Predict one image
```bash
python predict.py --model_path models/cat_dog_cnn.keras --image_path path/to/image.jpg
```

## Model highlights
- 3 convolution blocks
- Batch normalization
- Max pooling
- Dropout regularization
- Binary cross-entropy loss
- Adam optimizer
- Early stopping and learning-rate reduction

## Real-world applications
- Pet image sorting
- Visual quality checks
- Introductory computer vision projects
- Binary image classification demos
