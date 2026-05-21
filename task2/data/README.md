# Data folder

Place the raw Kaggle Dogs vs Cats images in `data/raw`.

Example:
```text
data/raw/
  cat.0.jpg
  cat.1.jpg
  dog.0.jpg
  dog.1.jpg
```

Then run `prepare_data.py` to split them into `data/processed/train` and `data/processed/val`.
