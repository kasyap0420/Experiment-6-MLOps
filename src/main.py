"""Main end-to-end ML pipeline entry point for Experiment 6.

Orchestrates: data loading -> cleaning -> feature engineering ->
training -> evaluation -> prediction -> saving artifacts.
"""
import json
import logging
import time
from pathlib import Path

import yaml

from data_loader import (inspect_dataset, load_dataset, save_inspection)
from evaluate import (compare_metrics, compute_metrics, load_saved_model,
                      save_metrics)
from feature_engineering import (engineer_features, save_feature_report)
from preprocessing import (clean_data, save_cleaning_report, split_data)
from predict import (load_model_artifact, predict, save_predictions)
from train import (build_pipeline, save_model, train_model)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("exp6-pipeline")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    # Resolve paths relative to the project, not the shell's working directory.
    for key, relative_path in config.get("paths", {}).items():
        config["paths"][key] = str((PROJECT_ROOT / relative_path).resolve())
    return config


def run_pipeline(config: dict, run_tag: str = "run_1") -> dict:
    """Execute the complete end-to-end ML pipeline."""
    t_start = time.perf_counter()
    paths = config["paths"]

    # --- Task 1: Dataset Preparation ---
    log.info("Task 1: Dataset Preparation")
    ds = load_dataset(source=config["dataset"]["source"])
    info = inspect_dataset(ds)
    log.info("Loaded dataset: %d samples, %d features", ds.n_samples, ds.n_features)
    save_inspection(info, f"{paths['outputs']}/{run_tag}_dataset_inspection.json")

    X, y, cleaning_report = clean_data(ds.X, ds.y)
    log.info("Cleaning report: %s", json.dumps(cleaning_report))
    save_cleaning_report(cleaning_report, f"{paths['outputs']}/{run_tag}_cleaning_report.json")

    split = split_data(
        X, y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=config.get("stratify", True),
    )
    log.info("Train/test split: %s / %s", split.X_train.shape, split.X_test.shape)

    # --- Feature Engineering ---
    log.info("Feature Engineering")
    result = engineer_features(
        split.X_train, split.X_test, split.y_train, split.y_test
    )
    save_feature_report(result.transformations,
                        f"{paths['outputs']}/{run_tag}_feature_report.json")
    log.info("Transformations: %s", result.transformations)

    # --- Task 3 & 4: Train ---
    log.info("Model Training")
    pipeline = build_pipeline(
        n_estimators=config["model"]["n_estimators"],
        random_state=config["model"]["random_state"],
        imputer_strategy=config["preprocessing"]["imputer_strategy"],
    )
    pipeline = train_model(pipeline, result.X_train, result.y_train)

    # --- Evaluate ---
    log.info("Model Evaluation")
    y_pred = pipeline.predict(result.X_test)
    y_proba = pipeline.predict_proba(result.X_test)[:, 1]
    metrics = compute_metrics(result.y_test, y_pred, y_proba)
    log.info("Evaluation metrics: %s", json.dumps(metrics))
    save_metrics(metrics, f"{paths['outputs']}/{run_tag}_metrics.json")

    # --- Save model ---
    metadata = {
        "version": "v1.0",
        "features": result.feature_names,
        "class_names": {0: "malignant", 1: "benign"},
        "model_type": "RandomForestPipeline",
        **metrics,
    }
    model_path = f"{paths['models']}/model.joblib"
    save_model(pipeline, model_path, metadata)
    log.info("Saved model to %s", model_path)

    # --- Task 5: Prediction on unseen data ---
    log.info("Prediction")
    artifact = load_model_artifact(model_path)
    predictions = predict(artifact, result.X_test)
    pred_path = f"{paths['outputs']}/{run_tag}_predictions.csv"
    save_predictions(predictions, pred_path)
    log.info("Saved %d predictions to %s", len(predictions), pred_path)

    t_total = time.perf_counter() - t_start
    log.info("Pipeline completed in %.4f seconds", t_total)

    return {
        "metrics": metrics,
        "predictions": predictions,
        "total_time_sec": round(t_total, 4),
        "model_path": model_path,
        "pred_path": pred_path,
    }


def run_reproducibility_check(config: dict) -> dict:
    """Run the pipeline twice and compare results (Task 5)."""
    log.info("=== Reproducibility Run 1 ===")
    result_1 = run_pipeline(config, run_tag="run_1")

    log.info("=== Reproducibility Run 2 ===")
    result_2 = run_pipeline(config, run_tag="run_2")

    comparison = compare_metrics(result_1["metrics"], result_2["metrics"])
    log.info("Reproducibility comparison: %s", json.dumps(comparison, indent=2))

    save_metrics(comparison, f"{config['paths']['outputs']}/reproducibility_comparison.json")

    return {
        "run_1": result_1,
        "run_2": result_2,
        "comparison": comparison,
    }


def main() -> None:
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    config = load_config(config_path)
    log.info("Configuration loaded from %s", config_path)

    reproducibility = run_reproducibility_check(config)
    log.info("Overall consistent: %s",
             reproducibility["comparison"]["overall_consistent"])
    log.info("End-to-end Experiment 6 pipeline complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("Experiment 6 pipeline failed")
        raise
