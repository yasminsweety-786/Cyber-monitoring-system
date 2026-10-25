"""
Scan Service: Core Orchestration Layer.
Integrates URL parsing, feature extraction, ML inference, SHAP/attribution explainability,
risk assessment, recommendation generation, and database logging.
"""

import os
import sys
import time
import joblib
from datetime import datetime
from typing import Dict, Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.config import (
    MODEL_FILE, PREPROCESSOR_FILE, EXPLAINER_FILE, 
    LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD
)
from backend.database import save_scan_record
from backend.services.network_inspector import inspect_network_safety
from ml.feature_extractor import extract_url_features, normalize_url
from ml.preprocessor import URLPreprocessor
from ml.explainer import ModelExplainer
from ml.risk_scorer import calculate_risk_assessment

class ScanService:
    _instance = None

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.explainer = None
        self.load_ml_components()

    @classmethod
    def get_instance(cls) -> "ScanService":
        if cls._instance is None:
            cls._instance = ScanService()
        return cls._instance

    def load_ml_components(self):
        """Loads or reloads serialized model, preprocessor, and explainer."""
        try:
            if os.path.exists(MODEL_FILE):
                self.model = joblib.load(MODEL_FILE)
            if os.path.exists(PREPROCESSOR_FILE):
                self.preprocessor = URLPreprocessor.load(PREPROCESSOR_FILE)
            self.explainer = ModelExplainer(
                model_path=MODEL_FILE,
                preprocessor_path=PREPROCESSOR_FILE,
                explainer_path=EXPLAINER_FILE
            )
        except Exception as e:
            print(f"Warning: Failed to load ML components ({e}). Model must be trained.")

    def scan_url(self, raw_url: str, include_network_probe: bool = False) -> Dict[str, Any]:
        """
        Full end-to-end scanning pipeline for a URL.
        """
        t0 = time.time()
        
        # 1. URL Normalization & Validation
        normalized = normalize_url(raw_url)
        if not normalized or len(normalized) < 4:
            raise ValueError("Invalid or empty URL provided.")

        # Ensure ML model is loaded
        if self.model is None or self.preprocessor is None:
            self.load_ml_components()
            if self.model is None or self.preprocessor is None:
                raise RuntimeError("ML model artifacts are not initialized. Please run 'python ml/train.py' first.")

        # 2. Extract Extensible Features
        extracted_features = extract_url_features(normalized, include_network=include_network_probe)
        
        # 3. Safe Network Probe if requested
        network_data = None
        if include_network_probe:
            network_data = inspect_network_safety(normalized)
            extracted_features["redirect_count"] = network_data.get("redirect_count", 0)

        # 4. Feature Preprocessing
        X_vec = self.preprocessor.transform(normalized)

        # 5. Model Inference
        # Get malicious class probability (Index 1)
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_vec)[0]
            prob_malicious = float(probs[1]) if len(probs) > 1 else float(probs[0])
        else:
            pred_raw = self.model.predict(X_vec)[0]
            prob_malicious = 1.0 if pred_raw == 1 else 0.0

        # 6. Risk Scoring & Recommendations
        risk_data = calculate_risk_assessment(
            prob_malicious, 
            low_threshold=LOW_RISK_THRESHOLD, 
            high_threshold=HIGH_RISK_THRESHOLD
        )

        # 7. Explainability
        explanation_data = self.explainer.explain_url(normalized, top_k=6)
        key_factors = explanation_data["top_factors"]
        explain_method = explanation_data["explainability_method"]

        execution_time_ms = round((time.time() - t0) * 1000, 2)
        timestamp_now = datetime.now().isoformat()

        # Build response payload
        is_malicious = risk_data["riskScore"] >= 50
        prediction_str = "Potentially Malicious / Phishing" if is_malicious else "Legitimate / Benign"

        response_dict = {
            "url": raw_url,
            "normalized_url": normalized,
            "prediction": prediction_str,
            "is_malicious": is_malicious,
            "riskLevel": risk_data["riskLevel"],
            "riskScore": risk_data["riskScore"],
            "modelProbability": risk_data["modelProbability"],
            "confidence": risk_data["confidence"],
            "color": risk_data["color"],
            "badge": risk_data["badge"],
            "keyFactors": key_factors,
            "extractedFeatures": extracted_features,
            "recommendation": risk_data["recommendation"],
            "actionAdvice": risk_data["actionAdvice"],
            "explainabilityMethod": explain_method,
            "executionTimeMs": execution_time_ms,
            "timestamp": timestamp_now,
            "disclaimer": risk_data["disclaimer"]
        }

        # 8. Persist into SQLite Scan History
        try:
            record_id = save_scan_record(response_dict)
            response_dict["id"] = record_id
        except Exception as e:
            print(f"Warning: Failed to save scan history ({e})")
            response_dict["id"] = None

        return response_dict
