
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

LABELS = {
    0: "Negative",
    1: "Positive",
    2: "Neutral"
}

def get_model(name: str):
    name = name.lower().strip()
    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=2000,
            random_state=42,
            class_weight="balanced",
            n_jobs=None,
        )
    if name == "naive_bayes":
        return MultinomialNB(alpha=0.3)
    if name == "svm":
        # Calibrated model gives predict_proba for confidence scores
        svm = LinearSVC(C=1.0, random_state=42)
        return CalibratedClassifierCV(svm, cv=2)
    raise ValueError(f"Unknown model: {name}")

def available_models():
    return ["logistic_regression", "naive_bayes", "svm"]
