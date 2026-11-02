"""Data loading module for Experiment 6 end-to-end ML pipeline."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer


@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series
    feature_names: list
    target_name: str
    n_samples: int
    n_features: int


def load_dataset(source: str = "sklearn") -> Dataset:
    """Load the Breast Cancer Wisconsin (Diagnostic) dataset from scikit-learn."""
    if source != "sklearn":
        raise ValueError(f"Unsupported dataset source: {source}")

    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target

    dataset = Dataset(
        X=X,
        y=y,
        feature_names=list(X.columns),
        target_name=data.target_names.tolist(),
        n_samples=int(X.shape[0]),
        n_features=int(X.shape[1]),
    )
    return dataset


def inspect_dataset(dataset: Dataset) -> dict:
    """Inspect the dataset and return a summary dictionary."""
    info = {
        "n_samples": dataset.n_samples,
        "n_features": dataset.n_features,
        "feature_names": dataset.feature_names,
        "target_name": dataset.target_name,
        "target_distribution": dataset.y.value_counts().to_dict(),
        "missing_values": int(dataset.X.isna().sum().sum()),
        "duplicate_rows": int(dataset.X.duplicated().sum()),
        "dtypes": {c: str(t) for c, t in dataset.X.dtypes.items()},
        "numeric_features": list(dataset.X.select_dtypes(include=[np.number]).columns),
        "categorical_features": list(dataset.X.select_dtypes(exclude=[np.number]).columns),
    }
    return info


def save_inspection(info: dict, path: str) -> None:
    """Save dataset inspection summary to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    ds = load_dataset()
    info = inspect_dataset(ds)
    print(json.dumps(info, indent=2))
