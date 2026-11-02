"""Evaluation module for Experiment 6 end-to-end ML pipeline."""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


def load_saved_model(path: str) -> dict:
    """Load the saved model artifact (pipeline + metadata)."""
    artifact = joblib.load(path)
    return artifact


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Compute all classification evaluation metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }
    return metrics


def save_metrics(metrics: dict, path: str) -> None:
    """Save evaluation metrics to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def compare_metrics(metrics_a: dict, metrics_b: dict, tolerance: float = 1e-6) -> dict:
    """Compare two metric dictionaries for reproducibility verification."""
    comparison = {}
    consistent = True
    for key in metrics_a:
        if key in metrics_b:
            diff = abs(metrics_a[key] - metrics_b[key])
            comparison[key] = {
                "run_1": metrics_a[key],
                "run_2": metrics_b[key],
                "difference": diff,
                "consistent": diff <= tolerance,
            }
            if diff > tolerance:
                consistent = False
    comparison["overall_consistent"] = consistent
    return comparison


if __name__ == "__main__":
    from feature_engineering import engineer_features
    from preprocessing import clean_data, split_data
    from data_loader import load_dataset
    from train import build_pipeline, train_model

    ds = load_dataset()
    X, y, _ = clean_data(ds.X, ds.y)
    split = split_data(X, y)
    result = engineer_features(split.X_train, split.X_test, split.y_train, split.y_test)

    pipeline = build_pipeline()
    pipeline = train_model(pipeline, result.X_train, result.y_train)
    y_pred = pipeline.predict(result.X_test)
    y_proba = pipeline.predict_proba(result.X_test)[:, 1]
    metrics = compute_metrics(result.y_test, y_pred, y_proba)
    print(json.dumps(metrics, indent=2))
