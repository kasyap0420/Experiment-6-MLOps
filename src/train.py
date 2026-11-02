"""Training module for Experiment 6 end-to-end ML pipeline."""
import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline(n_estimators: int = 200, random_state: int = 42,
                   imputer_strategy: str = "median") -> Pipeline:
    """Build the scikit-learn Pipeline: Imputer -> Scaler -> RandomForest."""
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=imputer_strategy)),
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)),
    ])
    return pipeline


def train_model(pipeline: Pipeline, X_train, y_train) -> Pipeline:
    """Fit the pipeline on training data."""
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(pipeline: Pipeline, X_test, y_test) -> dict:
    """Compute evaluation metrics on the test set."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }
    return metrics, y_pred, y_proba


def save_model(pipeline: Pipeline, path: str, metadata: dict) -> None:
    """Save the trained pipeline and metadata."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": pipeline,
        "metadata": metadata,
    }
    joblib.dump(artifact, path)


def save_metrics(metrics: dict, path: str) -> None:
    """Save evaluation metrics to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    from feature_engineering import engineer_features
    from preprocessing import clean_data, split_data
    from data_loader import load_dataset

    ds = load_dataset()
    X, y, _ = clean_data(ds.X, ds.y)
    split = split_data(X, y)
    result = engineer_features(split.X_train, split.X_test, split.y_train, split.y_test)

    pipeline = build_pipeline()
    pipeline = train_model(pipeline, result.X_train, result.y_train)
    metrics, y_pred, y_proba = evaluate_model(pipeline, result.X_test, result.y_test)
    print("Metrics:", json.dumps(metrics, indent=2))
