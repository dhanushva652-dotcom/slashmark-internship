from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_predict, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import ModelConfig
from .features import add_engineered_features, feature_columns
from .metrics import evaluate_predictions, expected_cost

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Class" not in df.columns:
        raise ValueError("CSV must contain a 'Class' column.")
    return df

def build_preprocessor(numeric_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                ImbPipeline(steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_cols,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

def build_candidate_pipelines(numeric_cols: list[str], random_state: int):
    preprocessor = build_preprocessor(numeric_cols)

    candidates = {
        "logistic_smote": ImbPipeline(steps=[
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=random_state, sampling_strategy=0.25)),
            ("under", RandomUnderSampler(random_state=random_state, sampling_strategy=0.5)),
            ("model", LogisticRegression(
                max_iter=4000,
                class_weight="balanced",
                solver="liblinear",
                random_state=random_state,
            )),
        ]),
        "random_forest_smote": ImbPipeline(steps=[
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=random_state, sampling_strategy=0.25)),
            ("under", RandomUnderSampler(random_state=random_state, sampling_strategy=0.5)),
            ("model", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=random_state,
            )),
        ]),
        "hist_gb_weighted": ImbPipeline(steps=[
            ("preprocess", preprocessor),
            ("model", HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_depth=6,
                max_iter=250,
                l2_regularization=0.1,
                random_state=random_state,
            )),
        ]),
        "decision_tree_weighted": ImbPipeline(steps=[
            ("preprocess", preprocessor),
            ("model", DecisionTreeClassifier(
                max_depth=8,
                min_samples_leaf=25,
                class_weight="balanced",
                random_state=random_state,
            )),
        ]),
    }
    return candidates

def choose_threshold(y_true, y_score, fraud_cost: float, false_alarm_cost: float):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    thresholds = np.append(thresholds, 1.0)

    best = {"threshold": 0.5, "f2": -1.0, "expected_cost": float("inf")}
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f2 = (5 * prec * rec / (4 * prec + rec)) if (4 * prec + rec) else 0.0
        cost = fp * false_alarm_cost + fn * fraud_cost
        if (f2 > best["f2"]) or (np.isclose(f2, best["f2"]) and cost < best["expected_cost"]):
            best = {"threshold": float(t), "f2": float(f2), "expected_cost": float(cost)}
    return best

def pick_best_model(X_train, y_train, candidates, random_state: int):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1", "recall": "recall"}

    rows = []
    for name, pipe in candidates.items():
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
        rows.append({
            "model": name,
            "roc_auc": float(np.mean(scores["test_roc_auc"])),
            "pr_auc": float(np.mean(scores["test_pr_auc"])),
            "f1": float(np.mean(scores["test_f1"])),
            "recall": float(np.mean(scores["test_recall"])),
        })
    results = pd.DataFrame(rows).sort_values(["pr_auc", "roc_auc", "recall"], ascending=False).reset_index(drop=True)
    best_name = results.loc[0, "model"]
    return best_name, results

def main():
    parser = argparse.ArgumentParser(description="Train a fraud detection model.")
    parser.add_argument("--data", type=Path, required=True, help="Path to creditcard.csv")
    parser.add_argument("--outdir", type=Path, default=Path("artifacts"), help="Output directory")
    args = parser.parse_args()

    cfg = ModelConfig()
    args.outdir.mkdir(parents=True, exist_ok=True)

    raw = load_data(args.data)
    data = add_engineered_features(raw)
    print(data.columns.tolist())
    target = data["Class"].astype(int)
    X = data.drop(columns=["Class"])

    feature_cols = feature_columns()
    X = X[feature_cols]

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.random_state)
    train_idx, test_idx = next(splitter.split(X, target))
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = target.iloc[train_idx].copy(), target.iloc[test_idx].copy()

    candidates = build_candidate_pipelines(list(X_train.columns), cfg.random_state)
    best_name, cv_table = pick_best_model(X_train, y_train, candidates, cfg.random_state)
    best_model = candidates[best_name]

    best_model.fit(X_train, y_train)

    # Out-of-fold scores for threshold tuning on training set.
    oof_scores = cross_val_predict(
        best_model,
        X_train,
        y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.random_state),
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    threshold_info = choose_threshold(
        y_train.to_numpy(),
        oof_scores,
        fraud_cost=cfg.fraud_cost,
        false_alarm_cost=cfg.false_alarm_cost,
    )
    threshold = threshold_info["threshold"]

    test_scores = best_model.predict_proba(X_test)[:, 1]
    test_pred = (test_scores >= threshold).astype(int)
    metrics = evaluate_predictions(
        y_test.to_numpy(),
        test_pred,
        test_scores,
        false_alarm_cost=cfg.false_alarm_cost,
        fraud_cost=cfg.fraud_cost,
    )

    # Save artifacts
    joblib.dump(best_model, args.outdir / "best_model.joblib")
    (args.outdir / "best_threshold.json").write_text(json.dumps({
        "threshold": threshold,
        "training_f2": threshold_info["f2"],
        "training_expected_cost": threshold_info["expected_cost"],
        "best_model": best_name,
    }, indent=2))

    (args.outdir / "cv_results.csv").write_text(cv_table.to_csv(index=False))
    (args.outdir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.outdir / "classification_report.txt").write_text(metrics["classification_report"])

    # ROC and PR curve data
    fpr, tpr, roc_thresholds = roc_curve(y_test, test_scores)
    (args.outdir / "roc_curve.csv").write_text(
        pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": np.append(roc_thresholds, np.nan)[:len(fpr)]}).to_csv(index=False)
    )

    precision, recall, pr_thresholds = precision_recall_curve(y_test, test_scores)
    (args.outdir / "pr_curve.csv").write_text(
        pd.DataFrame({"precision": precision, "recall": recall, "threshold": np.append(pr_thresholds, np.nan)[:len(precision)]}).to_csv(index=False)
    )

    summary = {
        "best_model": best_name,
        "threshold": threshold,
        "test_metrics": metrics,
        "cv_results": cv_table.to_dict(orient="records"),
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
