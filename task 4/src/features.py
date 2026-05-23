from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]

def validate_input_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df[REQUIRED_COLUMNS].copy()

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    original_df = df.copy()

    df = validate_input_frame(df).copy()

    amount = df["Amount"].astype(float)
    time = df["Time"].astype(float)

    df["amount_log1p"] = np.log1p(np.clip(amount, a_min=0, a_max=None))
    df["amount_sqrt"] = np.sqrt(np.clip(amount, a_min=0, a_max=None))
    df["time_sin"] = np.sin(2 * np.pi * (time % 86400) / 86400.0)
    df["time_cos"] = np.cos(2 * np.pi * (time % 86400) / 86400.0)
    df["time_hours"] = (time % 86400) / 3600.0
    df["time_amount_interaction"] = df["amount_log1p"] * df["time_sin"]

    pca_cols = [f"V{i}" for i in range(1, 29)]
    pca_abs = df[pca_cols].abs()

    df["pca_abs_mean"] = pca_abs.mean(axis=1)
    df["pca_abs_std"] = pca_abs.std(axis=1)
    df["pca_energy"] = (df[pca_cols] ** 2).sum(axis=1)
    df["pca_max_abs"] = pca_abs.max(axis=1)
    df["pca_min"] = df[pca_cols].min(axis=1)
    df["pca_max"] = df[pca_cols].max(axis=1)

    if "Class" in original_df.columns:
        df["Class"] = original_df["Class"]

    return df

def feature_columns() -> list[str]:
    base = REQUIRED_COLUMNS
    engineered = [
        "amount_log1p",
        "amount_sqrt",
        "time_sin",
        "time_cos",
        "time_hours",
        "time_amount_interaction",
        "pca_abs_mean",
        "pca_abs_std",
        "pca_energy",
        "pca_max_abs",
        "pca_min",
        "pca_max",
    ]
    return base + engineered