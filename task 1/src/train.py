
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from src.models import LABELS, available_models, get_model
from src.preprocessing import preprocess_data

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "sample_reviews_3class.csv"
MODELS_DIR = ROOT / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"
MODEL_PATH = MODELS_DIR / "best_sentiment_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"

def load_dataset(path=DATA_PATH):
    df = pd.read_csv(path)
    if "Review" not in df.columns or "Liked" not in df.columns:
        raise ValueError("Dataset must contain 'Review' and 'Liked' columns.")
    return df

def build_vectorizer():
    return TfidfVectorizer(
        max_features=4000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )

def train_and_select_best_model():
    df = load_dataset()
    df = preprocess_data(df)

    X = df["cleaned_review"]
    y = df["Liked"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    vectorizer = build_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    results = []
    best_model = None
    best_name = None
    best_f1 = -1.0
    best_pred = None

    for name in available_models():
        model = get_model(name)
        model.fit(X_train_vec, y_train)
        pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="weighted")
        results.append({
            "model": name,
            "accuracy": float(acc),
            "f1_weighted": float(f1),
        })
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name
            best_pred = pred

    metrics = {
        "best_model": best_name,
        "labels": LABELS,
        "results": results,
        "accuracy": float(accuracy_score(y_test, best_pred)),
        "f1_weighted": float(f1_score(y_test, best_pred, average="weighted")),
        "classification_report": classification_report(
            y_test,
            best_pred,
            target_names=[LABELS[i] for i in sorted(LABELS)]
        ),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Training completed")
    print(f"Best model: {best_name}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-score: {metrics['f1_weighted']:.4f}")
    print("\nClassification report:")
    print(metrics["classification_report"])
    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved vectorizer -> {VECTORIZER_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")

    return metrics

if __name__ == "__main__":
    train_and_select_best_model()
