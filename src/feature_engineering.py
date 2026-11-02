"""Feature engineering module for Experiment 6 end-to-end ML pipeline."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class FeatureEngineeringResult:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list
    transformations: list


def engineer_features(X_train: pd.DataFrame, X_test: pd.DataFrame,
                      y_train: pd.Series, y_test: pd.Series) -> FeatureEngineeringResult:
    """Apply feature engineering transformations."""
    transformations = []

    X_train = X_train.copy()
    X_test = X_test.copy()
    common_features = list(X_train.columns)
    X_test = X_test[common_features]

    for col in common_features:
        if not np.issubdtype(X_train[col].dtype, np.number):
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
            transformations.append(f"Converted {col} to numeric")

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    transformations.append("Cast all features to float32")

    result = FeatureEngineeringResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=common_features,
        transformations=transformations,
    )
    return result


def save_feature_report(transformations: list, path: str) -> None:
    """Save feature engineering report to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"transformations": transformations}, f, indent=2)


if __name__ == "__main__":
    from preprocessing import clean_data, split_data
    from data_loader import load_dataset

    ds = load_dataset()
    X, y, _ = clean_data(ds.X, ds.y)
    split = split_data(X, y)
    result = engineer_features(split.X_train, split.X_test, split.y_train, split.y_test)
    print("Feature engineering complete.")
    print("Transformations:", result.transformations)
    print(f"Train: {result.X_train.shape}, Test: {result.X_test.shape}")
