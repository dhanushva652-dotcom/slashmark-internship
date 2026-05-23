from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .features import add_engineered_features, feature_columns

def main():
    parser = argparse.ArgumentParser(description="Score a transaction with the fraud model.")
    parser.add_argument("--model", type=Path, required=True, help="Path to best_model.joblib")
    parser.add_argument("--input", type=Path, required=True, help="Path to a JSON file containing one transaction")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    args = parser.parse_args()

    model = joblib.load(args.model)
    raw = json.loads(args.input.read_text())
    df = pd.DataFrame([raw])
    df = add_engineered_features(df)
    df = df[feature_columns()]

    score = float(model.predict_proba(df)[:, 1][0])
    pred = int(score >= args.threshold)

    result = {
        "fraud_probability": score,
        "prediction": pred,
        "label": "fraud" if pred == 1 else "legit",
    }
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
