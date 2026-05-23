# Credit Card Fraud Detection AI

An end-to-end machine learning project for credit card fraud detection using:
- feature engineering
- imbalance handling with SMOTE and undersampling
- model comparison
- cross-validation
- threshold tuning
- cost-sensitive evaluation

## Dataset
Download the Kaggle dataset: **Credit Card Fraud Detection** (`creditcard.csv`).

Place the CSV here:
```bash
data/creditcard.csv
```

## Install
```bash
pip install -r requirements.txt
```

## Train
```bash
python train.py --data data/creditcard.csv --outdir artifacts
```

## Predict
```bash
python predict.py --model artifacts/best_model.joblib --input sample_transaction.json
```

## Outputs
The training script saves:
- trained model
- scaler/preprocessing pipeline
- evaluation report
- confusion matrix
- ROC and PR curves
- model card
- monitoring checklist

## Notes
- The dataset is highly imbalanced, so the project uses resampling plus cost-sensitive metrics.
- The model tunes the decision threshold for fraud detection, not just raw accuracy.
