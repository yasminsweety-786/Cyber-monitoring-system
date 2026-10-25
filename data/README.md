# Data Directory

This directory houses the raw and processed datasets utilized for training and evaluating the ML-Based Intelligent Cyber Monitoring System.

## Structure

- `raw/`: Raw CSV datasets.
  - `raw_urls_dataset.csv`: 30,596 balanced, deduplicated real-world URLs from Tranco Top 1M, PhishTank, URLhaus, and OpenPhish.
  - `dataset.csv`: Standard reference dataset.
- `processed/`: Stratified train/validation/test partitions.
  - `train.csv`: 70% stratified training partition (21,416 samples).
  - `val.csv`: 15% validation partition (4,590 samples).
  - `test.csv`: 15% held-out unseen test partition (4,590 samples).
- `cyber_monitor.db`: SQLite database storing real-time scan logs, history, and threat statistics.
- `dataset_info.json`: Metadata summary detailing records count, sources, and class distributions.

## Reproducibility

To re-fetch and generate the raw dataset:
```bash
python ml/build_raw_dataset.py
```
To inspect and verify dataset balance:
```bash
python ml/inspect_dataset.py
```
