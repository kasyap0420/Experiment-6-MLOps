# Experiment 6 - End-to-End ML Pipeline Implementation

This repository contains the implementation and recorded results of **Experiment 6: End-to-End ML Pipeline Implementation**.

## Aim

Build a reusable, executable, and reproducible machine-learning workflow covering dataset loading, validation and cleaning, preprocessing, feature engineering, model training, evaluation, saved-model prediction, and reporting.

## Dataset and model

- **Dataset:** Breast Cancer Wisconsin (Diagnostic), loaded with `sklearn.datasets.load_breast_cancer`. No separate dataset download or CSV is required.
- **Task:** Binary classification: `0 = malignant`, `1 = benign`.
- **Input:** 30 numerical features across 569 records.
- **Split:** 80% training, 20% test, stratified, with `random_state=42`.
- **Preprocessing:** Median imputation and standard scaling, fitted as part of the scikit-learn pipeline on training data.
- **Model:** Random Forest classifier with 200 trees and `random_state=42`.
- **Saved artifact:** `models/model.joblib` contains the fitted model pipeline and metadata.

## What was implemented

1. Dataset inspection, cleaning, train/test splitting, and feature preparation.
2. A modular Python pipeline that trains and evaluates the model without manual notebook steps.
3. Classification metrics: accuracy, precision, recall, F1-score, and ROC-AUC. In the saved evaluation, class `1` (benign) is the positive class for precision, recall, and F1-score.
4. Two runs using the same dataset and configuration, followed by a reproducibility comparison.
5. Reloading the saved model and exporting predictions for the 114 held-out test records in each run.
6. Six visualizations generated from the saved metrics, prediction CSVs, dataset inspection, and trained model; an architecture diagram; and a PDF report containing a genuine terminal screenshot.

The prediction CSVs do not include the original test labels or all class probabilities needed to reconstruct a genuine ground-truth confusion matrix or ROC curve. Neither is fabricated in this report.

## Project files

| Path | Purpose |
| --- | --- |
| `config/config.yaml` | Dataset, split, preprocessing, and model settings. |
| `src/data_loader.py` | Load and inspect the dataset. |
| `src/preprocessing.py` | Clean data and create the training/test split. |
| `src/feature_engineering.py` | Prepare and align numerical features. |
| `src/train.py` | Build, train, and save the scikit-learn pipeline. |
| `src/evaluate.py` | Calculate metrics and compare runs. |
| `src/predict.py` | Load the saved model and export predictions. |
| `src/main.py` | Run the complete workflow twice for reproducibility verification. |
| `models/model.joblib` | Saved model pipeline and metadata. |
| `outputs/` | Metrics, predictions, dataset reports, and reproducibility results. |
| `generate_evidence_plots.py` | Generate six figures using saved experimental data. |
| `generate_report.py` | Produce the PDF from verified outputs and figures. |
| `rebuild_evidence_and_report.py` | Regenerate figures, then the PDF, without retraining. |
| `report_assets/` | Experimental graphs, architecture diagram, terminal screenshot, and evidence manifest. |
| `Experiment_6_Report.pdf` | Academic experiment report. |
| `github_repository_url.txt` | Actual public repository URL used in the PDF report. |

## Run the experiment on Windows

Use **Windows Command Prompt (CMD)**. Every command below must be executed **inside the repository root** — the folder containing `src`, `config`, and `generate_report.py`. If you have just cloned the repository, open CMD in that folder (in File Explorer, open the folder, click the address bar, type `cmd`, and press Enter).

### 1. Verify Python and install required packages

Run these commands from the repository root:

```cmd
python --version
python -m pip install numpy pandas scikit-learn pyyaml joblib matplotlib pillow reportlab
```

### 2. Train, evaluate, predict, and verify reproducibility

Run from the repository root:

```cmd
python src\main.py
```

This creates or updates the trained model in `models/` and the recorded results in `outputs/`. The script performs two runs and compares their evaluation metrics.

### 3. Generate the six figures

Run from the repository root:

```cmd
python generate_evidence_plots.py
```

The figures are saved to `report_assets/`, using actual outputs and the saved model.

### 4. Generate the PDF report

The report requires both `report_assets\pipeline_diagram.png` and a **real terminal screenshot** saved as `report_assets\actual_terminal_screenshot.png` (capture it from a completed terminal execution). It also reads the actual public GitHub URL from `github_repository_url.txt`.

Run from the repository root:

```cmd
python generate_report.py
```

Or regenerate the six plots and report together, without retraining:

```cmd
python rebuild_evidence_and_report.py
```

The report is saved to `Experiment_6_Report.pdf`. The rebuild script checks required inputs and replaces the report only after successful generation.

## Reproducibility and limitations

- The pipeline uses a fixed random state and a stratified train/test split.
- Saved results include a comparison of two runs and prediction CSVs for both runs.
- The visualization script uses saved experimental outputs and the fitted Random Forest model; it does not generate dummy data.
- Feature importance indicates how the model used features; it does not establish causation.
- Only load the `.joblib` model artifact from a source you trust, because Python serialization may execute code during loading.
