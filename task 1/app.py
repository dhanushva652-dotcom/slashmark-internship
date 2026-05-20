
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from src.predict import predict_sentiment
from src.train import train_and_select_best_model

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)

MODEL_PATH = ROOT / "models" / "best_sentiment_model.pkl"
VECTORIZER_PATH = ROOT / "models" / "tfidf_vectorizer.pkl"

if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
    # Train automatically on first run
    train_and_select_best_model()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("review") or "").strip()
    if not text:
        return jsonify({"error": "Please enter a review or text."}), 400
    try:
        return jsonify(predict_sentiment(text))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

@app.route("/examples")
def examples():
    return jsonify({
        "examples": [
            "The food was amazing and the service was excellent!",
            "Terrible experience. Food was cold and the waiter was rude.",
            "The service was okay. Nothing special but not bad either.",
        ]
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
