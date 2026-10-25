"""
Dataset inspection and exploratory data analysis tool.
Inspects the raw URL dataset, displays columns, class distribution, missing values, duplicates, and writes summary metadata.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "raw_urls_dataset.csv")

def inspect_dataset(data_path=DEFAULT_PATH, output_json=True):
    print("=" * 65)
    print("      DATASET INSPECTION & REPRODUCIBILITY REPORT")
    print("=" * 65)
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return None
        
    df = pd.read_csv(data_path)
    
    filename = os.path.basename(data_path)
    total_rows = len(df)
    total_cols = len(df.columns)
    col_names = list(df.columns)
    
    print(f"Dataset File      : {filename}")
    print(f"File Path         : {data_path}")
    print(f"Total Records     : {total_rows:,}")
    print(f"Total Columns     : {total_cols}")
    print(f"Column Names      : {col_names}")
    print("-" * 65)
    
    # Missing values
    missing = df.isnull().sum()
    missing_dict = missing.to_dict()
    print("Missing Values:")
    if missing.sum() == 0:
        print("  None (0 missing values across all columns)")
    else:
        for col, cnt in missing[missing > 0].items():
            print(f"  {col}: {cnt:,} missing ({cnt/total_rows*100:.2f}%)")
    print("-" * 65)
    
    # Duplicates
    dup_count = int(df.duplicated(subset=['url'] if 'url' in df.columns else None).sum())
    print(f"Duplicate Records : {dup_count:,} ({dup_count/total_rows*100:.2f}%)")
    print("-" * 65)
    
    # Target column
    target_col = 'label' if 'label' in df.columns else ('phishing' if 'phishing' in df.columns else df.columns[-1])
    class_counts = df[target_col].value_counts().to_dict()
    class_pct = (df[target_col].value_counts(normalize=True) * 100).round(2).to_dict()
    
    print(f"Target Column     : '{target_col}'")
    print("Class Distribution:")
    for cls_val, cnt in class_counts.items():
        cls_name = "Legitimate / Benign (0)" if cls_val == 0 else "Malicious / Phishing (1)"
        print(f"  Class {cls_val} [{cls_name}]: {cnt:,} samples ({class_pct[cls_val]}%)")
    print("-" * 65)
    
    # Sources breakdown if present
    source_counts = {}
    if 'source' in df.columns:
        source_counts = df['source'].value_counts().to_dict()
        print("Data Source Breakdown:")
        for src, cnt in source_counts.items():
            print(f"  {src}: {cnt:,} samples ({cnt/total_rows*100:.2f}%)")
        print("-" * 65)
        
    # Sample URLs
    if 'url' in df.columns:
        print("Sample Benign URLs:")
        for u in df[df[target_col] == 0]['url'].head(3):
            print(f"  [+] {u}")
        print("\nSample Malicious/Phishing URLs:")
        for u in df[df[target_col] == 1]['url'].head(3):
            print(f"  [!] {u}")
        print("-" * 65)
        
    summary = {
        "dataset_name": "Cybersecurity Multi-Source URL Benchmark",
        "dataset_filename": filename,
        "source": "Tranco Top 1M, PhishTank Verified, URLhaus (abuse.ch), OpenPhish Live Feed",
        "total_records": total_rows,
        "target_column": target_col,
        "classes": {
            "0": "Legitimate / Benign",
            "1": "Malicious / Phishing"
        },
        "class_distribution": {
            "0": int(class_counts.get(0, 0)),
            "1": int(class_counts.get(1, 0))
        },
        "class_percentage": {
            "0": float(class_pct.get(0, 0.0)),
            "1": float(class_pct.get(1, 0.0))
        },
        "missing_values": int(missing.sum()),
        "duplicate_records": dup_count,
        "sources": source_counts
    }
    
    if output_json:
        out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        os.makedirs(out_dir, exist_ok=True)
        json_path = os.path.join(out_dir, "dataset_info.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved dataset metadata summary to: {json_path}")
        
    return summary

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    inspect_dataset(path)
