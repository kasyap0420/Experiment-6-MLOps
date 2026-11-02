#!/usr/bin/env python3
"""Build verified Experiment 6 figures ONLY from existing experimental artifacts.

Run:
    python generate_evidence_plots.py

Reads outputs/*.json, outputs/run_*_predictions.csv and models/model.joblib.
Does not train, generate labels, fabricate data, modify outputs or touch the PDF.
Use joblib only on a saved model you created/trust: deserialization can run code.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / 'outputs'
ASSETS = ROOT / 'report_assets'
MODELS = ROOT / 'models'
METRIC_KEYS = ('accuracy', 'precision', 'recall', 'f1', 'roc_auc')
DISPLAY = ('Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC')
PLOT_NAMES = (
    '01_dataset_class_balance.png',
    '02_run_metrics_comparison.png',
    '03_predicted_class_counts.png',
    '04_prediction_confidence_hist.png',
    '05_feature_importances.png',
    '06_run_to_run_agreement.png',
)
SOURCE_NAMES = (
    'outputs/run_1_dataset_inspection.json',
    'outputs/run_1_metrics.json',
    'outputs/run_2_metrics.json',
    'outputs/reproducibility_comparison.json',
    'outputs/run_1_predictions.csv',
    'outputs/run_2_predictions.csv',
    'models/model.joblib',
)
COLORS = ('#335D87', '#BE7445')


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f'Missing required real output: {path}')
    with path.open('r', encoding='utf-8') as f:
        result = json.load(f)
    if not isinstance(result, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return result


def require_int(value, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f'Invalid integer for {label}: {value!r}')
    n = int(value)
    if n < 0 or (isinstance(value, float) and value != n):
        raise ValueError(f'Invalid nonnegative integer for {label}: {value!r}')
    return n


def get_metrics(path: Path) -> dict:
    raw = load_json(path)
    for key in METRIC_KEYS:
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not (0 <= value <= 1):
            raise ValueError(f'{path.name} has missing/invalid metric: {key}={value!r}')
    return raw


def read_predictions(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f'Missing required genuine predictions: {path}')
    rows = []
    with path.open('r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        required = {'record_id', 'predicted_class', 'predicted_label', 'confidence'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Missing columns in {path}: {sorted(required - set(reader.fieldnames or []))}')
        seen = set()
        for line, raw in enumerate(reader, 2):
            rec_id = (raw.get('record_id') or '').strip()
            cls = (raw.get('predicted_class') or '').strip()
            name = (raw.get('predicted_label') or '').strip()
            if not rec_id or rec_id in seen or cls not in {'0','1'} or not name:
                raise ValueError(f'Invalid or duplicate prediction on line {line} in {path.name}')
            seen.add(rec_id)
            try:
                confidence = float(raw.get('confidence', ''))
            except ValueError as exc:
                raise ValueError(f'Invalid confidence on line {line} in {path.name}') from exc
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError(f'Confidence outside [0,1] on line {line} in {path.name}')
            rows.append({'record_id': rec_id, 'class': int(cls), 'name': name, 'confidence': confidence})
    if not rows:
        raise ValueError(f'No saved predictions in {path}')
    return rows


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def setup_fig(width: float = 7.5, height: float = 3.65):
    fig = Figure(figsize=(width, height), dpi=180, facecolor='white', constrained_layout=True)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_facecolor('white')
    return fig, ax


def finish(fig, path: Path):
    fig.savefig(path, dpi=180, facecolor='white')
    if path.stat().st_size < 2500:
        raise ValueError(f'Generated image looks empty: {path}')


def plot_class_balance(info: dict, names: list[str], out: Path) -> None:
    counts = [require_int(info['target_distribution'].get(str(i), info['target_distribution'].get(i)), f'class {i}') for i in (0, 1)]
    if sum(counts) != require_int(info['n_samples'], 'n_samples'):
        raise ValueError('Observed class distribution does not match dataset record count')
    fig, ax = setup_fig()
    bars = ax.bar(names, counts, color=COLORS, width=0.48)
    ax.set_title('Actual dataset target-class distribution', fontweight='bold')
    ax.set_ylabel('Dataset records')
    ax.set_ylim(0, max(counts) * 1.22)
    ax.bar_label(bars, padding=4, fontsize=10)
    ax.grid(axis='y', alpha=.15)
    ax.set_axisbelow(True)
    finish(fig, out)


def plot_metrics(a: dict, b: dict, out: Path) -> None:
    fig, ax = setup_fig(8.1, 3.8)
    x = list(range(len(METRIC_KEYS)))
    p = ax.bar([i-.18 for i in x], [a[k] for k in METRIC_KEYS], .36, label='Run 1', color=COLORS[0])
    q = ax.bar([i+.18 for i in x], [b[k] for k in METRIC_KEYS], .36, label='Run 2', color=COLORS[1])
    ax.set_xticks(x, DISPLAY)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Saved metric score (0 to 1)')
    ax.set_title('Held-out test metrics: two recorded runs', fontweight='bold')
    ax.legend(loc='lower right', frameon=False)
    ax.grid(axis='y', alpha=.15)
    ax.set_axisbelow(True)
    ax.bar_label(p, labels=[f'{a[k]:.3f}' for k in METRIC_KEYS], padding=3, fontsize=8)
    ax.bar_label(q, labels=[f'{b[k]:.3f}' for k in METRIC_KEYS], padding=3, fontsize=8)
    finish(fig, out)


def plot_prediction_counts(rows: list[dict], names: list[str], out: Path) -> None:
    counts = [sum(r['class'] == i for r in rows) for i in (0, 1)]
    fig, ax = setup_fig()
    bars = ax.bar(names, counts, color=COLORS, width=.46)
    ax.bar_label(bars, padding=5)
    ax.set_ylim(0, max(counts) * 1.22)
    ax.set_ylabel('Predicted test records')
    ax.set_title(f'Predicted classes on the held-out test set (n={len(rows)})', fontweight='bold')
    ax.grid(axis='y', alpha=.15)
    ax.set_axisbelow(True)
    finish(fig, out)


def plot_confidence(rows: list[dict], names: list[str], out: Path) -> None:
    fig, ax = setup_fig(8.1, 3.8)
    bins = [i / 20 for i in range(21)]
    for i in (0, 1):
        values = [r['confidence'] for r in rows if r['class'] == i]
        if values:
            ax.hist(values, bins=bins, alpha=.62, label=f'{names[i]} (n={len(values)})', color=COLORS[i], edgecolor='white')
    ax.set_title('Saved prediction confidence by predicted class', fontweight='bold')
    ax.set_xlabel('Probability of the assigned class (recorded confidence)')
    ax.set_ylabel('Test records per 0.05 confidence interval')
    ax.set_xlim(0, 1)
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=.13)
    ax.set_axisbelow(True)
    finish(fig, out)


def plot_importances(artifact: dict, info: dict, out: Path) -> None:
    pipe = artifact.get('pipeline')
    metadata = artifact.get('metadata') or {}
    model = getattr(pipe, 'named_steps', {}).get('model')
    importance = getattr(model, 'feature_importances_', None)
    features = metadata.get('features') or info.get('feature_names')
    if importance is None or not isinstance(features, list) or len(features) != len(importance):
        raise ValueError('Saved model has no compatible feature_importances_ / feature names; cannot plot without fabricating data')
    vals = [float(v) for v in importance]
    if not all(math.isfinite(v) and v >= 0 for v in vals):
        raise ValueError('Invalid feature importance values in saved trained model')
    ranked = sorted(zip(features, vals), key=lambda p: p[1], reverse=True)[:12]
    ranked.reverse()
    fig, ax = setup_fig(8.3, 4.35)
    bars = ax.barh([str(p[0]) for p in ranked], [p[1] for p in ranked], color=COLORS[0])
    ax.set_xlabel('Trained Random Forest impurity-based importance')
    ax.set_title('Top 12 recorded trained-model feature importances', fontweight='bold')
    ax.grid(axis='x', alpha=.13)
    ax.set_axisbelow(True)
    ax.bar_label(bars, labels=[f'{p[1]:.3f}' for p in ranked], padding=3, fontsize=8)
    ax.set_xlim(0, max(p[1] for p in ranked) * 1.25)
    finish(fig, out)


def plot_run_agreement(first: list[dict], second: list[dict], names: list[str], out: Path) -> list[list[int]]:
    ids1 = [r['record_id'] for r in first]
    ids2 = [r['record_id'] for r in second]
    if ids1 != ids2:
        raise ValueError('Run 1 and Run 2 prediction record_id/order differ; cannot plot aligned agreement')
    cells = [[0, 0], [0, 0]]
    for a, b in zip(first, second):
        cells[a['class']][b['class']] += 1
    fig, ax = setup_fig(7.0, 4.0)
    ax.imshow(cells, cmap='Blues', vmin=0, vmax=max(1, max(max(row) for row in cells)))
    ax.set_xticks([0,1], names)
    ax.set_yticks([0,1], names)
    ax.set_xlabel('Run 2 saved predicted class')
    ax.set_ylabel('Run 1 saved predicted class')
    ax.set_title('Run-to-run prediction agreement (NOT a confusion matrix)', fontweight='bold')
    for i in (0,1):
        for j in (0,1):
            ax.text(j, i, str(cells[i][j]), ha='center', va='center', fontsize=15,
                    color='white' if cells[i][j] > max(max(row) for row in cells) / 2 else '#1F2F44')
    finish(fig, out)
    return cells


def load_and_validate() -> dict:
    for filename in SOURCE_NAMES:
        if not (ROOT / filename).is_file():
            raise FileNotFoundError(f'Missing actual experimental source: {ROOT / filename}')
    info = load_json(OUTPUTS / 'run_1_dataset_inspection.json')
    for key in ('n_samples', 'n_features', 'target_distribution', 'feature_names', 'target_name'):
        if key not in info:
            raise ValueError(f'Dataset inspection missing {key}')
    names = info['target_name']
    if not isinstance(names, list) or len(names) != 2:
        raise ValueError('Requires binary target class names from actual inspection JSON')
    metrics1 = get_metrics(OUTPUTS / 'run_1_metrics.json')
    metrics2 = get_metrics(OUTPUTS / 'run_2_metrics.json')
    comparison = load_json(OUTPUTS / 'reproducibility_comparison.json')
    for key in METRIC_KEYS:
        actual_delta = abs(metrics1[key] - metrics2[key])
        entry = comparison.get(key)
        if not isinstance(entry, dict) or abs(float(entry.get('difference', float('nan'))) - actual_delta) > 1e-9 or bool(entry.get('consistent')) != (actual_delta <= 1e-6):
            raise ValueError(f'Inconsistent reproducibility record for {key}')
    if comparison.get('overall_consistent') is not all(abs(metrics1[k] - metrics2[k]) <= 1e-6 for k in METRIC_KEYS):
        raise ValueError('overall_consistent contradicts saved metrics')
    preds1 = read_predictions(OUTPUTS / 'run_1_predictions.csv')
    preds2 = read_predictions(OUTPUTS / 'run_2_predictions.csv')
    if len(preds1) != len(preds2):
        raise ValueError('Run prediction counts differ')
    for rows in (preds1, preds2):
        for r in rows:
            if r['name'] != str(names[r['class']]):
                raise ValueError(f'Predicted label disagrees with saved dataset class mapping: {r}')
    # The saved joblib is trusted because it was produced by the user's own training pipeline.
    # Never load arbitrary downloaded model artifacts: joblib/pickle can run code.
    import joblib
    artifact = joblib.load(MODELS / 'model.joblib')
    if not isinstance(artifact, dict) or 'pipeline' not in artifact:
        raise ValueError('Unexpected saved model artifact shape')
    return {'inspection':info, 'names':names, 'metrics1':metrics1, 'metrics2':metrics2,
            'comparison':comparison, 'preds1':preds1, 'preds2':preds2, 'artifact':artifact}


def generate() -> None:
    data = load_and_validate()   # no files altered if evidence is absent or invalid
    ASSETS.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='._exp6_figures_', dir=ASSETS))
    try:
        plot_class_balance(data['inspection'], data['names'], stage / PLOT_NAMES[0])
        plot_metrics(data['metrics1'], data['metrics2'], stage / PLOT_NAMES[1])
        plot_prediction_counts(data['preds1'], data['names'], stage / PLOT_NAMES[2])
        plot_confidence(data['preds1'], data['names'], stage / PLOT_NAMES[3])
        plot_importances(data['artifact'], data['inspection'], stage / PLOT_NAMES[4])
        cells = plot_run_agreement(data['preds1'], data['preds2'], data['names'], stage / PLOT_NAMES[5])
        manifest = {
            'evidence_policy': 'Existing saved Experiment 6 outputs and fitted model only; no synthetic or assumed values',
            'sources_sha256': {filename: sha256(ROOT / filename) for filename in SOURCE_NAMES},
            'figures': list(PLOT_NAMES),
            'n_dataset': int(data['inspection']['n_samples']),
            'n_predictions_run_1': len(data['preds1']),
            'n_predictions_run_2': len(data['preds2']),
            'class_distribution': data['inspection']['target_distribution'],
            'agreement_matrix': cells,
            'positive_class_for_metrics': 'benign (class 1)',
            'limitations': 'Prediction CSV does not contain ground-truth test labels or both class probabilities; therefore no confusion matrix or ROC/PR curve was constructed.',
        }
        (stage / 'evidence_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        for filename in PLOT_NAMES + ('evidence_manifest.json',):
            os.replace(stage / filename, ASSETS / filename)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    print('REAL EVIDENCE PLOTS GENERATED: 6')
    print('SOURCE: saved experiment JSON/CSV and own trained model.joblib (no dummy data)')
    for name in PLOT_NAMES:
        print('  ' + str(ASSETS / name))
    print('Evidence manifest: ' + str(ASSETS / 'evidence_manifest.json'))


if __name__ == '__main__':
    try:
        generate()
    except Exception as exc:
        print(f'EVIDENCE GENERATION FAILED: {exc}', file=sys.stderr)
        raise SystemExit(1)
