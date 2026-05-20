
from pathlib import Path

import joblib
import numpy as np

from src.models import LABELS
from src.preprocessing import clean_text

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_sentiment_model.pkl"
VECTORIZER_PATH = ROOT / "models" / "tfidf_vectorizer.pkl"

def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

def label_to_sentiment(label: int) -> str:
    return LABELS.get(int(label), "Unknown")

def predict_sentiment(text: str):
    model, vectorizer = load_artifacts()
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])

    prediction = int(model.predict(vector)[0])
    sentiment = label_to_sentiment(prediction)

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vector)[0]
        confidence = float(np.max(proba))
    elif hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(vector))
        confidence = float(np.max(scores))
    else:
        confidence = 0.0

    return {
        "prediction": prediction,
        "sentiment": sentiment,
        "confidence": round(confidence, 4),
        "cleaned_text": cleaned,
    }
