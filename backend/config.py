"""
Backend application configuration and directory paths.
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATABASE_PATH = os.path.join(PROJECT_ROOT, "data", "cyber_monitor.db")

MODEL_FILE = os.path.join(MODELS_DIR, "best_model.joblib")
PREPROCESSOR_FILE = os.path.join(MODELS_DIR, "preprocessor.joblib")
EXPLAINER_FILE = os.path.join(MODELS_DIR, "explainer.joblib")
METADATA_FILE = os.path.join(MODELS_DIR, "model_metadata.json")
EVALUATION_FILE = os.path.join(MODELS_DIR, "evaluation_results.json")
DATASET_INFO_FILE = os.path.join(DATA_DIR, "dataset_info.json")

# Configurable Risk Thresholds
LOW_RISK_THRESHOLD = int(os.getenv("LOW_RISK_THRESHOLD", "40"))
HIGH_RISK_THRESHOLD = int(os.getenv("HIGH_RISK_THRESHOLD", "70"))

# Network Probe Safety Timeout (seconds)
NETWORK_TIMEOUT_SECONDS = float(os.getenv("NETWORK_TIMEOUT_SECONDS", "2.5"))
