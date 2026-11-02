#!/usr/bin/env python3
"""Generate the Experiment 6 academic report from saved, verified pipeline outputs.

Run from any working directory:
    python generate_report.py

The script does not retrain, delete logs, or create execution screenshots.
It requires genuine user-supplied terminal/code screenshots and checks saved
reproducibility evidence before creating the report.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
from pathlib import Path
import tempfile
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
from PIL import Image as PILImage, UnidentifiedImageError
from reportlab.lib import colors
from generate_evidence_plots import PLOT_NAMES, SOURCE_NAMES, sha256
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus.doctemplate import LayoutError
from reportlab.platypus import (
    KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
    Table, TableStyle, Image as ReportImage,
)

LOG = logging.getLogger("exp6-report")
EXPERIMENT_TITLE = "END-TO-END ML PIPELINE IMPLEMENTATION"
EXPERIMENT_DATE = "16-10-2026"
STUDENT_NAME = "Y.V.R. Kasyap"
STUDENT_REG = "231FA04E47"
STUDENT_SECTION = "8"
METRIC_KEYS = ("accuracy", "precision", "recall", "f1", "roc_auc")
METRIC_NAMES = {
    "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
    "f1": "F1-score", "roc_auc": "ROC-AUC",
}
REQUIRED_PREDICTION_COLS = {
    "record_id", "predicted_class", "predicted_label", "confidence",
    "prediction_timestamp", "model_version",
}
NAVY = colors.HexColor("#23314B")
TEXT = colors.HexColor("#263244")
MUTED = colors.HexColor("#5E6B7B")
LIGHT = colors.HexColor("#EFF2F6")
RULE = colors.HexColor("#C9D1DB")


def read_repository_url(root: Path) -> str:
    """Use only the repository URL recorded after creating it on GitHub."""
    path = root / "github_repository_url.txt"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing GitHub repository URL file: {path}. "
            "Create the public repository first and save its verified HTTPS URL in this file."
        )
    url = path.read_text(encoding="utf-8-sig").strip()
    if not re.fullmatch(r"https://github\.com/[A-Za-z0-9-]+/Experiment-6-MLOps/?", url):
        raise ValueError(
            f"Invalid repository URL in {path}: {url!r}. "
            "Use the actual GitHub URL from `gh repo view`, not a placeholder."
        )
    return url.rstrip("/")


def read_json(path: Path, *, required_keys: tuple[str, ...] = ()) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required pipeline output: {path}")
    try:
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing JSON field '{key}' in {path}")
    return data


def read_config(path: Path) -> dict:
    """Read the experiment configuration, without duplicating ML settings."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing Experiment 6 configuration: {path}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required: python -m pip install pyyaml") from exc
    try:
        with path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read YAML configuration {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML configuration structure: {path}")
    for key in ("dataset", "test_size", "random_state", "preprocessing", "model"):
        if key not in config:
            raise ValueError(f"Missing configuration value '{key}' in {path}")
    return config


def require_valid_image(path: Path, label: str) -> tuple[int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    try:
        with PILImage.open(path) as im:
            im.verify()
        with PILImage.open(path) as im:
            width, height = im.size
            if width < 250 or height < 100:
                raise ValueError(f"{label} is too small to be readable: {width} x {height}")
            if im.format not in ("PNG", "JPEG"):
                raise ValueError(f"{label} must be PNG or JPEG, got {im.format}")
            return width, height
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Cannot open {label} at {path}: {exc}") from exc


def read_predictions(path: Path) -> tuple[int, list[dict], list[tuple[str, str, float]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing prediction CSV: {path}")
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            headers = set(reader.fieldnames or ())
            missing = REQUIRED_PREDICTION_COLS - headers
            if missing:
                raise ValueError(f"Missing CSV columns in {path}: {', '.join(sorted(missing))}")
            count, sample, signature = 0, [], []
            for count, row in enumerate(reader, 1):
                if not row["record_id"] or row["predicted_class"] not in ("0", "1"):
                    raise ValueError(f"Invalid record ID or predicted class at row {count + 1}: {path}")
                confidence = float(row["confidence"])
                if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                    raise ValueError(f"Invalid confidence at row {count + 1}: {path}")
                signature.append((row["record_id"], row["predicted_class"], confidence))
                if len(sample) < 2:
                    sample.append(row)
        if count < 2:
            raise ValueError(f"Prediction CSV requires at least 2 rows: {path}")
        return count, sample, signature
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Cannot read predictions from {path}: {exc}") from exc


def collect_evidence(root: Path, screenshot_override: Path | None) -> dict:
    outputs = root / "outputs"
    report_assets = root / "report_assets"
    # Verify submission artifacts without loading arbitrary joblib/pickle code.
    needed = [root / "models" / "model.joblib"] + [
        root / "src" / filename for filename in (
            "data_loader.py", "preprocessing.py", "feature_engineering.py",
            "train.py", "evaluate.py", "predict.py", "main.py",
        )
    ]
    absent = [str(p) for p in needed if not p.is_file()]
    if absent:
        raise FileNotFoundError("Missing model or pipeline modules: " + ", ".join(absent))
    config = read_config(root / "config" / "config.yaml")
    metrics = []
    for tag in ("run_1", "run_2"):
        values = read_json(outputs / f"{tag}_metrics.json", required_keys=METRIC_KEYS)
        for name in METRIC_KEYS:
            score = values[name]
            if isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"Invalid metric '{name}' in {tag}_metrics.json: {score!r}")
        metrics.append(values)
    first, second = metrics
    comparison = read_json(outputs / "reproducibility_comparison.json", required_keys=METRIC_KEYS + ("overall_consistent",))
    tolerance = 1e-6  # Matches this experiment's evaluation module.
    for key in METRIC_KEYS:
        entry = comparison[key]
        if not isinstance(entry, dict) or not all(k in entry for k in ("run_1", "run_2", "difference", "consistent")):
            raise ValueError(f"Invalid comparison data for '{key}'")
        delta = abs(first[key] - second[key])
        if any(abs(float(entry[k]) - value) > 1e-9 for k, value in (
            ("run_1", first[key]), ("run_2", second[key]), ("difference", delta)
        )) or entry["consistent"] != (delta <= tolerance):
            raise ValueError(f"Reproducibility comparison does not match saved metrics: {key}")
    actual_consistent = all(abs(first[k] - second[k]) <= tolerance for k in METRIC_KEYS)
    if comparison["overall_consistent"] is not actual_consistent:
        raise ValueError("'overall_consistent' disagrees with saved metric values")

    inspection = read_json(outputs / "run_1_dataset_inspection.json", required_keys=("n_samples", "n_features", "target_distribution"))
    cleaning = read_json(outputs / "run_1_cleaning_report.json", required_keys=("missing_values_before", "duplicate_rows_removed", "final_n_samples"))
    feature = read_json(outputs / "run_1_feature_report.json", required_keys=("transformations",))
    count1, sample, signature1 = read_predictions(outputs / "run_1_predictions.csv")
    count2, _sample2, signature2 = read_predictions(outputs / "run_2_predictions.csv")
    predictions_consistent = (
        count1 == count2 and
        all(a[:2] == b[:2] and abs(a[2] - b[2]) <= 1e-9
            for a, b in zip(signature1, signature2))
    )
    if count1 != count2:
        raise ValueError(f"Run prediction counts differ: {count1} vs {count2}")
    diagram = report_assets / "pipeline_diagram.png"
    screenshot = (screenshot_override or report_assets / "actual_terminal_screenshot.png").resolve()
    code_screenshot = (report_assets / "actual_code_screenshot.png").resolve()
    if not code_screenshot.is_file():
        raise FileNotFoundError(
            f"Missing genuine source-code screenshot: {code_screenshot}. "
            "Open src/train.py in VS Code, show the build_pipeline() function, "
            "capture that window with Win+Shift+S, and save it as "
            "report_assets/actual_code_screenshot.png. "
            "Do not make or substitute a simulated code screenshot."
        )
    require_valid_image(diagram, "pipeline diagram")
    require_valid_image(screenshot, "genuine terminal screenshot")
    require_valid_image(code_screenshot, "genuine source-code screenshot")
    manifest = read_json(report_assets / 'evidence_manifest.json', required_keys=('sources_sha256', 'figures'))
    if manifest['figures'] != list(PLOT_NAMES):
        raise ValueError('Evidence manifest does not match required genuine plot list; rerun generate_evidence_plots.py')
    for name in SOURCE_NAMES:
        if manifest['sources_sha256'].get(name) != sha256(root / name):
            raise ValueError(f'Underlying experimental output changed after plots were made: {name}. Rerun generate_evidence_plots.py')
    for filename in PLOT_NAMES:
        require_valid_image(report_assets / filename, f'genuine experimental figure {filename}')
    return {
        "config": config, "metrics1": first, "metrics2": second,
        "comparison": comparison, "inspection": inspection, "cleaning": cleaning,
        "feature": feature, "pred_count": count1, "pred_sample": sample,
        "predictions_consistent": predictions_consistent,
        "diagram": diagram, "screenshot": screenshot,
        "code_screenshot": code_screenshot,
    }


def fitted_image(path: Path, max_width: float, max_height: float) -> ReportImage:
    with PILImage.open(path) as im:
        width, height = im.size
    scale = min(max_width / width, max_height / height, 1.0)
    return ReportImage(str(path), width=width * scale, height=height * scale)


def draw_cover(canvas, doc) -> None:
    """Formal academic first page: precise, centered typesetting."""
    width, height = A4
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.85)
    canvas.line(0.95 * inch, height - 0.85 * inch, width - 0.95 * inch, height - 0.85 * inch)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 10)
    canvas.drawCentredString(width / 2, height - 1.55 * inch, "LABORATORY EXPERIMENT REPORT")
    canvas.setFillColor(NAVY)
    canvas.setFont("Times-Bold", 21)
    canvas.drawCentredString(width / 2, height - 2.42 * inch, "EXPERIMENT 6")
    canvas.setLineWidth(1.5)
    canvas.setStrokeColor(NAVY)
    canvas.line(width / 2 - 0.75 * inch, height - 2.73 * inch,
                width / 2 + 0.75 * inch, height - 2.73 * inch)
    canvas.setFont("Times-Bold", 19)
    canvas.drawCentredString(width / 2, height - 3.22 * inch, "END-TO-END ML PIPELINE")
    canvas.drawCentredString(width / 2, height - 3.58 * inch, "IMPLEMENTATION")

    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    left, right = 1.48 * inch, width - 1.48 * inch
    top, bottom = height - 4.37 * inch, height - 6.88 * inch
    canvas.roundRect(left, bottom, right - left, top - bottom, 9, stroke=1, fill=0)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawCentredString(width / 2, height - 4.72 * inch, "STUDENT DETAILS")
    canvas.setFillColor(TEXT)
    rows = [
        ("Name", STUDENT_NAME),
        ("Registration Number", STUDENT_REG),
        ("Section", STUDENT_SECTION),
    ]
    y = height - 5.14 * inch
    for label, value in rows:
        if label == "Section":
            # Keep the exact phrase requested, on a single line on page 1.
            canvas.setFont("Times-Bold", 14)
            canvas.drawCentredString(width / 2, y - 0.11 * inch, f"Section: {value}")
        else:
            canvas.setFont("Times-Roman", 12)
            canvas.drawCentredString(width / 2, y, label)
            canvas.setFont("Times-Bold", 14)
            canvas.drawCentredString(width / 2, y - 0.25 * inch, value)
        y -= 0.58 * inch
    canvas.setFont("Times-Bold", 12)
    canvas.drawCentredString(width / 2, height - 7.55 * inch, f"Date: {EXPERIMENT_DATE}")

    # Verified public GitHub URL, centered and clickable on the cover page.
    repository_url = doc.repository_url
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawCentredString(width / 2, height - 8.12 * inch, "PUBLIC GITHUB REPOSITORY")
    canvas.setFillColor(NAVY)
    canvas.setFont("Helvetica", 8.5)
    url_y = height - 8.37 * inch
    canvas.drawCentredString(width / 2, url_y, repository_url)
    link_width = canvas.stringWidth(repository_url, "Helvetica", 8.5)
    canvas.linkURL(repository_url,
                   (width / 2 - link_width / 2, url_y - 3,
                    width / 2 + link_width / 2, url_y + 10), relative=0)
    canvas.setStrokeColor(RULE)
    canvas.line(0.95 * inch, 0.92 * inch, width - 0.95 * inch, 0.92 * inch)
    canvas.setFillColor(MUTED)
    canvas.setFont("Times-Italic", 9)
    canvas.drawCentredString(width / 2, 0.67 * inch, "Experiment 6 - Machine Learning Operations")
    canvas.restoreState()


def draw_content_page(canvas, doc) -> None:
    canvas.saveState()
    w, h = A4
    canvas.setStrokeColor(RULE)
    canvas.line(doc.leftMargin, h - 0.58 * inch, w - doc.rightMargin, h - 0.58 * inch)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(doc.leftMargin, h - 0.46 * inch, "EXPERIMENT 6  |  END-TO-END ML PIPELINE")
    canvas.line(doc.leftMargin, 0.57 * inch, w - doc.rightMargin, 0.57 * inch)
    canvas.setFont("Times-Roman", 9)
    canvas.drawString(doc.leftMargin, 0.40 * inch, "Y.V.R. Kasyap  |  231FA04E47")
    canvas.drawRightString(w - doc.rightMargin, 0.40 * inch, f"Page {doc.page}")
    canvas.restoreState()


def esc(value) -> str:
    return escape(str(value))


def text_style(name: str, *, size: float = 10, leading: float = 14, bold: bool = False, after: int = 5) -> ParagraphStyle:
    return ParagraphStyle(
        name, parent=getSampleStyleSheet()["BodyText"],
        fontName="Times-Bold" if bold else "Times-Roman",
        fontSize=size, leading=leading, textColor=TEXT, spaceAfter=after,
        alignment=TA_LEFT, allowWidows=0, allowOrphans=0,
    )


def make_table(rows: list[list[str]], widths: list[float], style: ParagraphStyle) -> Table:
    formatted = [[Paragraph(esc(value), style) for value in row] for row in rows]
    table = Table(formatted, colWidths=widths, hAlign="CENTER", repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def build_report(root: Path, evidence: dict, output_file: Path) -> None:
    repository_url = read_repository_url(root)
    cfg = evidence["config"]
    values = evidence["metrics1"]
    run2 = evidence["metrics2"]
    comparison = evidence["comparison"]
    info = evidence["inspection"]
    cleaned = evidence["cleaning"]
    transforms = evidence["feature"].get("transformations", [])
    assets = root / "report_assets"
    assets.mkdir(parents=True, exist_ok=True)
    chart = assets / PLOT_NAMES[1]  # Both saved runs; sourced from metrics JSON only.

    body = text_style("Exp6Body", size=10, leading=14)
    heading = ParagraphStyle(
        "Exp6Heading", parent=body, fontName="Times-Bold", fontSize=13,
        leading=16, textColor=NAVY, spaceBefore=14, spaceAfter=8, keepWithNext=1,
    )
    small = text_style("Exp6Small", size=8.3, leading=11.5, after=0)
    caption = ParagraphStyle(
        "Exp6Caption", parent=body, fontName="Times-Italic", fontSize=8.5,
        leading=11.5, textColor=MUTED, alignment=TA_CENTER, spaceAfter=8,
    )
    right = A4[0] - 0.80 * inch * 2 - 12  # available inside the document frame
    story = [Spacer(1, 1), PageBreak()]

    def para(content: str):
        story.append(Paragraph(content, body))

    def head(name: str):
        story.append(Paragraph(name, heading))

    def figure(path: Path, max_height: float, title: str):
        story.append(KeepTogether([
            fitted_image(path, right, max_height),
            Spacer(1, 3), Paragraph(esc(title), caption),
        ]))

    head("Aim")
    para(
        "To convert a machine learning workflow into a reusable, executable and "
        "reproducible end-to-end ML pipeline covering data preparation, preprocessing, "
        "feature engineering, model training, evaluation and prediction."
    )

    head("Task 1 - Dataset Preparation")
    para(
        f"<b>Dataset:</b> Breast Cancer Wisconsin (Diagnostic), loaded through scikit-learn; "
        f"{int(info['n_samples'])} records and {int(info['n_features'])} numerical input features. "
        f"Target classes: 0 = malignant, 1 = benign. "
        f"Class distribution: {esc(info['target_distribution'])}."
    )
    para(
        f"<b>Data validation and cleaning:</b> Missing feature values before cleaning: "
        f"{int(cleaned['missing_values_before'])}; duplicate rows removed: "
        f"{int(cleaned['duplicate_rows_removed'])}; retained samples: "
        f"{int(cleaned['final_n_samples'])}. "
        f"No categorical encoding was required for this dataset."
    )
    test_size = float(cfg["test_size"])
    stratified = bool(cfg.get("stratify", False))
    para(
        f"<b>Split:</b> {(1-test_size):.0%} training and {test_size:.0%} testing; "
        f"stratification {'enabled' if stratified else 'disabled'}; "
        f"random_state = {esc(cfg['random_state'])}."
    )

    head("Task 2 - Design of the End-to-End Pipeline")
    para(
        "<b>Workflow:</b> Data loading to validation/cleaning to feature engineering "
        "to model training to evaluation to prediction. "
        "The trained model and transformations are stored together for consistent inference."
    )
    transforms_text = "; ".join(str(s) for s in transforms) or "No transformations documented"
    para(
        f"<b>Feature engineering:</b> {esc(transforms_text)}. "
        "The fitted SimpleImputer and StandardScaler are inside the scikit-learn Pipeline, "
        "so they learn parameters only from training data."
    )
    figure(evidence["diagram"], 2.45 * inch, "Figure 1. End-to-end ML pipeline architecture")
    para("<b>Stage inputs and outputs:</b>")
    stages = [
        ["Stage", "Input", "Output"],
        ["Data loading", "scikit-learn dataset", "Features X and labels y"],
        ["Cleaning", "Raw X, y", "Cleaned dataset, quality report"],
        ["Train/test split", "Cleaned X, y", "Training and held-out test sets"],
        ["Feature engineering", "Split feature sets", "Aligned numerical features"],
        ["Training", "Train features, labels, settings", "Fitted and saved Pipeline"],
        ["Evaluation", "Fitted Pipeline and test set", "Metrics JSON"],
        ["Prediction", "Saved Pipeline and test features", "Prediction CSV"],
    ]
    story.append(make_table(stages, [1.32*inch, 2.23*inch, 2.62*inch], small))

    head("Task 3 - Modular Pipeline Implementation")
    para("The experiment is organized into independent, reusable Python modules:")
    modules = [
        ["Module", "Purpose"],
        ["data_loader.py", "Load dataset; inspect features, target and dataset quality"],
        ["preprocessing.py", "Clean dataset and perform train/test split"],
        ["feature_engineering.py", "Align features, convert numeric types, save transformation summary"],
        ["train.py", "Build and train the saved scikit-learn Pipeline"],
        ["evaluate.py", "Calculate classification metrics and compare repeated runs"],
        ["predict.py", "Reload saved model; generate and export predictions"],
        ["main.py", "Orchestrate all stages and the reproducibility check"],
    ]
    story.append(make_table(modules, [1.52 * inch, 4.65 * inch], small))
    story.append(Spacer(1, 8))
    model_cfg = cfg["model"]
    preprocess_cfg = cfg["preprocessing"]
    para(
        f"<b>Configuration:</b> config/config.yaml; model: RandomForestClassifier "
        f"({esc(model_cfg.get('n_estimators', 'unspecified'))} estimators; "
        f"random_state = {esc(model_cfg.get('random_state', 'unspecified'))}); "
        f"imputation: {esc(preprocess_cfg.get('imputer_strategy', 'unspecified'))}; "
        f"scaling: {esc(preprocess_cfg.get('scaler', 'unspecified'))}. "
        "The model and fitted preprocessing pipeline are stored in models/model.joblib."
    )

    # Real implementation evidence, captured by the student from the source file.
    figure(evidence["code_screenshot"], 2.25 * inch,
           "Figure 2. Genuine VS Code screenshot of the train.py Pipeline implementation")

    # Separate Task 4 so all evaluation rows have enough room on one page.
    story.append(PageBreak())
    head("Task 4 - Execution and Model Evaluation")
    para(
        "The full workflow is executed with <b>python src/main.py</b> from the exp-6 "
        "directory. Saved results from two runs are available in outputs/. "
        "The figure below is the supplied terminal execution screenshot."
    )
    figure(evidence["screenshot"], 2.20 * inch, "Figure 3. Genuine terminal execution evidence")
    metrics_rows = [["Metric", "Run 1 value", "Meaning"]]
    meanings = {
        "accuracy": "Overall proportion classified correctly",
        "precision": "Precision for benign (class 1)",
        "recall": "Recall for benign (class 1)",
        "f1": "F1 for benign (class 1)",
        "roc_auc": "Discrimination using probability of class 1",
    }
    for key in METRIC_KEYS:
        metrics_rows.append([METRIC_NAMES[key], f"{values[key]:.4f}", meanings[key]])
    # One inseparable table: the ROC-AUC row cannot be orphaned on another page.
    story.append(KeepTogether([
        make_table(metrics_rows, [1.20 * inch, 1.05 * inch, 3.92 * inch], small),
        Spacer(1, 7),
    ]))
    figure(chart, 2.30 * inch, "Figure 4. Recorded evaluation metrics for Runs 1 and 2")

    # Start the reproducibility results on a fresh page; do not split its table.
    story.append(PageBreak())
    head("Task 5 - Reproducibility and Prediction")
    para(
        f"Two runs used the saved dataset and configuration with random_state = "
        f"{esc(cfg['random_state'])}. "
        f"Metric consistency: <b>{'Passed' if comparison['overall_consistent'] else 'Not passed'}</b>. "
        f"Prediction class and confidence consistency: "
        f"<b>{'Passed' if evidence['predictions_consistent'] else 'Not passed'}</b>."
    )
    repeated = [["Metric", "Run 1", "Run 2", "Absolute difference", "Match"]]
    for name in METRIC_KEYS:
        d = comparison[name]
        repeated.append([
            METRIC_NAMES[name], f"{values[name]:.6f}", f"{run2[name]:.6f}",
            f"{d['difference']:.6f}", "Yes" if d["consistent"] else "No",
        ])
    story.append(KeepTogether([
        make_table(repeated, [1.10*inch, 1.04*inch, 1.04*inch, 1.36*inch, 0.66*inch], small)
    ]))
    story.append(Spacer(1, 7))
    para(
        f"<b>Prediction:</b> {evidence['pred_count']} held-out test records were predicted "
        "using the reloaded saved pipeline. Outputs contain record IDs, predicted classes, "
        "labels, confidences, timestamps and model versions."
    )
    sample_rows = [["Record ID", "Class", "Label", "Confidence", "Model version"]]
    for r in evidence["pred_sample"]:
        sample_rows.append([
            r["record_id"], r["predicted_class"], r["predicted_label"],
            r["confidence"], r["model_version"],
        ])
    story.append(make_table(sample_rows, [1.05*inch, .66*inch, 1.12*inch, 1.12*inch, 1.15*inch], small))

    # Do not strand the Task 6 heading at the bottom of a crowded page.
    story.append(PageBreak())
    head("Task 6 - Documentation, Observations and Result")
    para(
        f"<b>Public GitHub repository:</b> "
        f'<link href="{esc(repository_url)}" color="#23314B">{esc(repository_url)}</link>.'
    )
    para(
        "<b>Project structure:</b> exp-6/config/ (settings); exp-6/src/ (pipeline modules); "
        "exp-6/models/ (saved pipeline); exp-6/outputs/ (run metrics, CSV predictions "
        "and comparison); exp-6/report_assets/ (diagram, chart and terminal screenshot); "
        "exp-6/generate_report.py (PDF generation)."
    )
    para(
        f"<b>Observations:</b> {int(info['n_samples'])} records and "
        f"{int(info['n_features'])} input features were processed. "
        f"Both runs saved {evidence['pred_count']} predictions. "
        f"All five evaluation metrics {'matched' if comparison['overall_consistent'] else 'did not match'} "
        "between the saved runs. The report generator verifies required files and checks "
        "the saved comparison before issuing this report."
    )
    log_file = root / "outputs" / "pipeline_run.log"
    if log_file.is_file():
        para(
            "<b>Execution log:</b> outputs/pipeline_run.log is available for inspection "
            "of runtime messages and any encountered errors."
        )
    else:
        para("<b>Execution log:</b> no standalone pipeline_run.log was included in the supplied project copy.")
    head("Actual Issues Encountered and Corrective Actions")
    para(
        "The verified final ML runs completed successfully. The following issues "
        "occurred during project setup and report preparation, not as unverified "
        "model-training failures:"
    )
    issue_rows = [
        ["Observed issue", "Corrective action"],
        ["ReportLab was missing when the maintenance dependency check ran.",
         "Installed ReportLab in the Python environment; report generation then succeeded."],
        ["The report could not finalize when the required genuine terminal screenshot was absent.",
         "Captured and saved the actual terminal screenshot in report_assets."],
        ["VS Code reported eight unresolved ReportLab imports while Python 3.12 was selected.",
         "Selected the installed Python 3.14 interpreter; the eight warnings disappeared."],
        ["An earlier report split the evaluation-metrics table across two pages.",
         "Updated ReportLab pagination to keep all five metric rows together."],
    ]
    story.append(make_table(issue_rows, [2.85 * inch, 3.32 * inch], small))
    story.append(Spacer(1, 7))
    para(
        "<b>Result:</b> The reusable end-to-end ML pipeline covers dataset loading, "
        "cleaning, feature engineering, training, evaluation, saved-model inference "
        "and documented reproducibility checks."
    )

    # Visual evidence appendix: every plot is derived from exact saved experiment files.
    # No test ground-truth labels exist in the predictions.csv, so no fabricated
    # confusion matrix, ROC curve or precision-recall curve is included.
    story.append(PageBreak())
    head("Evidence Figures - Dataset and Inference")
    para(
        "The following figures were constructed exclusively from the saved experiment "
        "JSON/CSV outputs and fitted Random Forest model. Original data source filenames and "
        "SHA-256 fingerprints are recorded in <b>report_assets/evidence_manifest.json</b>. "
        "No fabricated values or simulated screenshots have been used."
    )
    figure(assets / PLOT_NAMES[0], 2.8 * inch,
           "Figure 5. Actual target-class distribution across all original dataset records")
    para(
        "Figure 5 uses outputs/run_1_dataset_inspection.json; these are actual "
        "dataset target labels, not model predictions."
    )
    figure(assets / PLOT_NAMES[2], 2.8 * inch,
           "Figure 6. Predicted classes for the held-out test records")
    para(
        "Figure 6 uses outputs/run_1_predictions.csv. Predicted-class totals are not "
        "the same as ground-truth class totals or classification correctness."
    )
    story.append(PageBreak())
    head("Evidence Figures - Confidence and Features")
    figure(assets / PLOT_NAMES[3], 2.8 * inch,
           "Figure 7. Distribution of saved prediction confidence scores")
    para(
        "Each bar represents an interval of confidence values read from the saved prediction CSV, "
        "grouped by the predicted class. Confidence is not a measure of correctness."
    )
    figure(assets / PLOT_NAMES[4], 3.0 * inch,
           "Figure 8. Top 12 impurity-based feature importances from the fitted model")
    para(
        "These feature importances are extracted from the trained Random Forest "
        "inside models/model.joblib. They do not establish causal relationships."
    )
    story.append(PageBreak())
    head("Evidence Figures - Reproducibility")
    figure(assets / PLOT_NAMES[5], 3.0 * inch,
           "Figure 9. Cross-run agreement between saved prediction classes")
    para(
        "The matrix compares Run 1 predictions with Run 2 predictions using aligned "
        "record IDs; it is <b>not a confusion matrix against ground-truth test labels</b>."
    )
    para(
        "<b>Evidence limitation:</b> The saved prediction CSVs lack true test labels and the "
        "positive-class probability for every record. Accordingly, this report does not show "
        "a ground-truth confusion matrix, per-threshold ROC curve or precision-recall curve. "
        "Those graphs would require saving the original test labels and probabilities "
        "during a separately documented evaluation run."
    )
    head("Evidence and Artifact Traceability")
    para(
        "<b>Experimental source files:</b> outputs/run_1_dataset_inspection.json; "
        "outputs/run_1_metrics.json; outputs/run_2_metrics.json; "
        "outputs/reproducibility_comparison.json; outputs/run_1_predictions.csv; "
        "outputs/run_2_predictions.csv; models/model.joblib. "
        "<b>Figure reproducibility:</b> python generate_evidence_plots.py. "
        "<b>Report regeneration:</b> python generate_report.py."
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix="._exp6_report_", suffix=".pdf", dir=str(output_file.parent))
    os.close(fd)
    try:
        doc = SimpleDocTemplate(
            temp_name, pagesize=A4, leftMargin=0.80*inch, rightMargin=0.80*inch,
            topMargin=0.85*inch, bottomMargin=0.78*inch,
            title="Experiment 6 - End-to-End ML Pipeline Implementation",
            author=STUDENT_NAME, subject=f"Student report - {STUDENT_REG}",
            pageCompression=1,
        )
        doc.repository_url = repository_url
        doc.build(story, onFirstPage=draw_cover, onLaterPages=draw_content_page)
        if not Path(temp_name).is_file() or Path(temp_name).stat().st_size < 1500:
            raise RuntimeError("ReportLab did not produce a valid nonempty PDF")
        try:
            os.replace(temp_name, output_file)
        except PermissionError as exc:
            raise PermissionError(f"Cannot overwrite {output_file}. Close it in your PDF viewer first.") from exc
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent,
                        help="Experiment 6 folder (default: script's folder)")
    parser.add_argument("--screenshot", type=Path, help="Path to actual terminal screenshot PNG/JPEG")
    parser.add_argument("--output", type=Path, help="Output PDF path (default: <project-root>/Experiment_6_Report.pdf)")
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    screenshot = args.screenshot.expanduser().resolve() if args.screenshot else None
    output = args.output.expanduser().resolve() if args.output else root / "Experiment_6_Report.pdf"
    try:
        if not root.is_dir():
            raise NotADirectoryError(f"Experiment 6 project directory not found: {root}")
        info = collect_evidence(root, screenshot)
        build_report(root, info, output)
    except (OSError, ValueError, RuntimeError, KeyError, LayoutError) as exc:
        LOG.error("Report generation failed: %s", exc)
        return 1
    print(f"REPORT GENERATED: {output}")
    print('VERIFIED VISUAL FIGURES IN PDF: 6 (plus pipeline diagram, real code screenshot and real terminal screenshot)')
    print(f"Student: {STUDENT_NAME} | Reg. No.: {STUDENT_REG} | Section: {STUDENT_SECTION}")
    print(f"Experiment date: {EXPERIMENT_DATE}")
    print(f"Public repository: {read_repository_url(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
