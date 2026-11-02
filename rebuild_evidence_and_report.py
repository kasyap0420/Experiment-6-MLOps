#!/usr/bin/env python3
"""Safe one-command replacement of Experiment 6 visual evidence and PDF.

No ML training, output mutation, or simulated evidence. Existing PDF is
replaced atomically ONLY after the new PDF is successfully built.
"""
from pathlib import Path
import sys

import generate_evidence_plots as visuals
import generate_report as report


def main() -> int:
    root = Path(__file__).resolve().parent
    expected = [
        root / 'config' / 'config.yaml',
        root / 'report_assets' / 'pipeline_diagram.png',
        root / 'report_assets' / 'actual_terminal_screenshot.png',
        root / 'outputs' / 'run_1_metrics.json',
        root / 'outputs' / 'run_1_predictions.csv',
        root / 'models' / 'model.joblib',
    ]
    missing = [str(p) for p in expected if not p.is_file()]
    if missing:
        raise FileNotFoundError('Project missing required genuine input(s):\n' + '\n'.join(missing))
    print('STEP 1/3: Creating six plots from genuine saved outputs and trained model...')
    visuals.generate()
    print('STEP 2/3: Building the replacement academic PDF...')
    existing_argv = sys.argv[:]
    try:
        sys.argv = [str(root / 'generate_report.py')]
        status = report.main()
    finally:
        sys.argv = existing_argv
    if status != 0:
        raise RuntimeError('PDF generator failed. Original report remains in place.')
    pdf = root / 'Experiment_6_Report.pdf'
    if not pdf.is_file() or pdf.stat().st_size < 2500:
        raise RuntimeError('Missing or suspiciously small new report PDF')
    with pdf.open('rb') as stream:
        if stream.read(5) != b'%PDF-':
            raise ValueError('Generated file does not have a PDF header')
    print('STEP 3/3: Removing only outdated generated assets...')
    for filename in ('metrics_chart.png', 'terminal_main.png'):
        stale = root / 'report_assets' / filename
        if stale.is_file():
            stale.unlink()
            print(f'Removed outdated asset: {stale}')
    print('\nSUCCESS: Replaced old PDF after a complete, validated rebuild.')
    print('FINAL PDF:', pdf)
    print('FIGURES:', root / 'report_assets')
    print('6 saved evidence graphs + architecture diagram + genuine terminal screenshot')
    print('Existing ML training results/models were NOT changed.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'REBUILD FAILED: {exc}', file=sys.stderr)
        raise SystemExit(1)
