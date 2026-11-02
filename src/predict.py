"""Prediction module for Experiment 6 end-to-end ML pipeline."""
import csv
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def load_model_artifact(path: str) -> dict:
    """Load the saved model artifact."""
    artifact = joblib.load(path)
    return artifact


def predict(artifact: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions and confidence scores using the saved pipeline."""
    pipeline = artifact["pipeline"]
    metadata = artifact.get("metadata", {})

    preds = pipeline.predict(X)
    proba = pipeline.predict_proba(X)
    confidence = proba.max(axis=1)

    class_names = metadata.get("class_names", {0: "malignant", 1: "benign"})
    feature_names = metadata.get("features", list(X.columns))

    output = pd.DataFrame({
        "record_id": [f"REC{str(i+1).zfill(4)}" for i in range(len(X))],
        "predicted_class": preds.astype(int),
        "predicted_label": [class_names[int(p)] for p in preds],
        "confidence": np.round(confidence, 4),
        "prediction_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_version": metadata.get("version", "v1.0"),
    })
    return output


def save_predictions(output: pd.DataFrame, path: str) -> None:
    """Save predictions to a CSV file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)


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

    artifact = {
        "pipeline": pipeline,
        "metadata": {
            "version": "v1.0",
            "features": result.feature_names,
            "class_names": {0: "malignant", 1: "benign"},
        },
    }

    output = predict(artifact, result.X_test)
    print(output.head())
    print(f"Total predictions: {len(output)}")
