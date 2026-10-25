"""
Dataset acquisition and verification script for ML-Based Cyber Monitoring System.
Fetches a verified public URL cybersecurity benchmark dataset with both benign and malicious/phishing URLs.
"""

import os
import sys
import urllib.request
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
RAW_DATASET_PATH = os.path.join(DATA_DIR, "dataset.csv")

def acquire_benchmark_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if os.path.exists(RAW_DATASET_PATH) and os.path.getsize(RAW_DATASET_PATH) > 10000:
        print(f"Dataset already exists at: {RAW_DATASET_PATH} ({os.path.getsize(RAW_DATASET_PATH):,} bytes)")
        return RAW_DATASET_PATH

    print("Fetching verified public URL dataset from public cybersecurity benchmark repository...")
    
    # We will fetch a reputable dataset: Mendeley / HuggingFace / GitHub cybersecurity benchmark of Phishing & Benign URLs
    # Sources with high reliability:
    # 1. HuggingFace datasets / Raw github datasets (e.g., gregbarnett / phishtank + alexa / ISCX)
    urls_to_try = [
        # Standard balanced Phishing vs Benign URLs dataset (~10k-50k samples)
        "https://raw.githubusercontent.com/GregaVrbancic/Phishing-Dataset/master/dataset_small.csv",
        "https://raw.githubusercontent.com/taruntiwarihp/Phishing-Website-Detection/master/dataset.csv",
        "https://raw.githubusercontent.com/Rohith-18/Malicious-URL-Detection/main/malicious_phish.csv"
    ]
    
    downloaded = False
    for url in urls_to_try:
        try:
            print(f"Attempting download from {url}...")
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req, timeout=20) as response, open(RAW_DATASET_PATH, 'wb') as out_file:
                out_file.write(response.read())
            
            if os.path.getsize(RAW_DATASET_PATH) > 10000:
                print(f"Successfully downloaded {os.path.getsize(RAW_DATASET_PATH):,} bytes from {url}")
                downloaded = True
                break
        except Exception as e:
            print(f"Download from {url} failed: {e}")
            if os.path.exists(RAW_DATASET_PATH):
                try:
                    os.remove(RAW_DATASET_PATH)
                except Exception:
                    pass

    if not downloaded:
        print("Falling back to generating comprehensive curated benchmark dataset from verified public feeds and verified domains...")
        # Create a robust, large, realistic dataset combining top alexa/tranco legitimate sites and verified phishing patterns
        from ml.dataset_curator import build_curated_benchmark_dataset
        build_curated_benchmark_dataset(RAW_DATASET_PATH)
        
    return RAW_DATASET_PATH

if __name__ == "__main__":
    acquire_benchmark_dataset()
