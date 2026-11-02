"""Preprocessing module for Experiment 6 end-to-end ML pipeline."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class PreprocessedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def clean_data(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Handle missing values, duplicates, and inconsistent data types."""
    X = X.copy()
    y = y.copy()

    missing_before = int(X.isna().sum().sum())

    mask_missing = X.isna().any(axis=1)
    X = X.loc[~mask_missing]
    y = y.loc[~mask_missing]

    dup_mask = X.duplicated(keep="first")
    X = X.loc[~dup_mask]
    y = y.loc[~dup_mask]

    for col in X.columns:
        if not np.issubdtype(X[col].dtype, np.number):
            X[col] = pd.to_numeric(X[col], errors="coerce")

    y = y.loc[X.index]

    missing_after = int(X.isna().sum().sum())
    duplicates_removed = int(dup_mask.sum())

    cleaning_report = {
        "missing_values_before": missing_before,
        "missing_values_after": missing_after,
        "duplicate_rows_removed": int(duplicates_removed),
        "final_n_samples": int(len(X)),
    }
    return X, y, cleaning_report


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2,
               random_state: int = 42, stratify: bool = True) -> PreprocessedData:
    """Split dataset into training and testing sets."""
    stratify_arg = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
    )
    return PreprocessedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


def save_cleaning_report(report: dict, path: str) -> None:
    """Save cleaning report to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    from data_loader import load_dataset, inspect_dataset

    ds = load_dataset()
    X, y, report = clean_data(ds.X, ds.y)
    print("Cleaning report:", json.dumps(report, indent=2))
    split = split_data(X, y)
    print(f"Train: {split.X_train.shape}, Test: {split.X_test.shape}")
