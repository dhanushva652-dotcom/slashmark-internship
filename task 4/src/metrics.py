from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

def expected_cost(y_true, y_pred, false_alarm_cost: float = 1.0, fraud_cost: float = 25.0) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return fp * false_alarm_cost + fn * fraud_cost

def evaluate_predictions(y_true, y_pred, y_score, false_alarm_cost: float = 1.0, fraud_cost: float = 25.0):
    return {
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": fbeta_score(y_true, y_pred, beta=1, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "expected_cost": expected_cost(y_true, y_pred, false_alarm_cost, fraud_cost),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }
