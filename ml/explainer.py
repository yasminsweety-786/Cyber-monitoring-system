"""
Explainability Module for ML-Based Cyber Monitoring System.
Computes local feature contributions for an individual URL prediction using SHAP (TreeExplainer)
with fallback to weighted feature importance attribution.
Translates technical feature signals into clear, human-understandable cybersecurity explanations.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.feature_extractor import extract_url_features
from ml.preprocessor import URLPreprocessor

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Human-readable cybersecurity descriptions mapped to features
FEATURE_DESCRIPTIONS = {
    "num_dots": "Excessive number of dots ({val}) often indicates subdomain deception or obfuscation.",
    "num_subdomains": "Elevated count of subdomains ({val}) mimicking legitimate brand names.",
    "has_ip_address": "Raw IP address host detected ({val}) instead of a verified registered domain name.",
    "suspicious_keyword_count": "Security-sensitive keywords ({val} found) such as login/verify/account in URL.",
    "has_suspicious_keyword": "Presence of targeted authentication/security keywords in URL path or parameters.",
    "url_length": "Abnormally long URL ({val} characters) commonly used for token hiding or obfuscation.",
    "hostname_length": "Long hostname ({val} chars) exceeding standard legitimate domain conventions.",
    "digit_ratio": "High proportion of numerical digits ({val:.1%}) in URL structure.",
    "num_digits": "High digit count ({val}) in hostname or path, typical of auto-generated phishing targets.",
    "domain_entropy": "High domain character entropy ({val:.2f}) indicating randomized or algorithmically generated domain (DGA).",
    "url_entropy": "High overall URL entropy ({val:.2f}) indicating encrypted/encoded payload parameters.",
    "is_https": "Unencrypted HTTP protocol used ({val}) - lacks TLS/SSL encryption.",
    "num_slashes": "Excessive path depth ({val} slashes) indicating deeply nested redirect or fake portal.",
    "num_params": "Multiple URL query parameters ({val}) attempting parameter stuffing.",
    "has_double_slash_in_path": "Double slash '//' redirect pattern found within URL path.",
    "is_suspicious_tld": "Top-Level Domain belongs to high-abuse/low-reputation registrar categories.",
    "num_at": "Commercial '@' symbol present in URL, which can disguise destination domain in some browsers.",
    "has_port": "Non-standard network port specified in host address.",
    "is_punycode": "Punycode (xn--) domain detected, often associated with IDN homograph spoofing.",
    "has_client_server_in_domain": "Keywords like 'server'/'client'/'admin' embedded in domain name."
}

class ModelExplainer:
    """Provides local and global feature attribution for URL risk predictions."""
    def __init__(self, model_path: Optional[str] = None, preprocessor_path: Optional[str] = None, explainer_path: Optional[str] = None):
        self.model_path = model_path or os.path.join(MODELS_DIR, "best_model.joblib")
        self.preprocessor_path = preprocessor_path or os.path.join(MODELS_DIR, "preprocessor.joblib")
        self.explainer_path = explainer_path or os.path.join(MODELS_DIR, "explainer.joblib")
        
        self.model = None
        self.preprocessor: Optional[URLPreprocessor] = None
        self.shap_explainer = None
        self.method_used = "SHAP TreeExplainer"
        self._load_artifacts()

    def _load_artifacts(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        if os.path.exists(self.preprocessor_path):
            self.preprocessor = URLPreprocessor.load(self.preprocessor_path)
            
        if os.path.exists(self.explainer_path):
            try:
                self.shap_explainer = joblib.load(self.explainer_path)
                self.method_used = "SHAP TreeExplainer (Exact Game-Theoretic Shapley Values)"
            except Exception as e:
                self.shap_explainer = None
                self.method_used = "Feature Importance & Gradient Attribution (Fallback)"
        else:
            self.method_used = "Feature Contribution Attribution (Fallback)"

    def explain_url(self, raw_url: str, top_k: int = 6) -> Dict[str, Any]:
        """
        Computes the primary factors contributing to the risk score for a single URL.
        Returns sorted list of contributing factors, SHAP values (or attribution weights),
        and human-readable cybersecurity context.
        """
        if self.model is None or self.preprocessor is None:
            self._load_artifacts()
            if self.model is None or self.preprocessor is None:
                raise RuntimeError("ML model and Preprocessor must be trained and loaded before explaining predictions.")

        raw_features = extract_url_features(raw_url)
        X_vec = self.preprocessor.transform(raw_url)
        feature_names = self.preprocessor.get_feature_names_out()
        
        factors = []
        
        # 1. Attempt SHAP local explanation
        if self.shap_explainer is not None:
            try:
                shap_values = self.shap_explainer.shap_values(X_vec)
                # For binary classification with XGBoost/TreeExplainer:
                # shap_values can be 1D array of length num_features or 2D
                if isinstance(shap_values, list):
                    # List for classes [class 0, class 1]
                    vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
                elif len(shap_values.shape) == 2:
                    vals = shap_values[0]
                elif len(shap_values.shape) == 3:
                    vals = shap_values[0, :, 1]
                else:
                    vals = np.array(shap_values).flatten()
                
                # Pair with feature names and raw values
                for idx, name in enumerate(feature_names):
                    val = vals[idx] if idx < len(vals) else 0.0
                    raw_val = raw_features.get(name, 0)
                    factors.append({
                        "feature": name,
                        "raw_value": raw_val,
                        "impact_score": float(val),
                        "abs_impact": abs(float(val)),
                        "direction": "increases_risk" if val > 0 else "decreases_risk"
                    })
                method = "SHAP TreeExplainer"
            except Exception as e:
                factors = self._compute_importance_attribution(X_vec, raw_features, feature_names)
                method = "Feature Weight Attribution (Documented Fallback)"
        else:
            factors = self._compute_importance_attribution(X_vec, raw_features, feature_names)
            method = "Feature Weight Attribution (Documented Fallback)"

        # Sort factors by impact magnitude
        factors.sort(key=lambda x: x["abs_impact"], reverse=True)
        top_factors = factors[:top_k]

        # Generate human-readable cybersecurity insights
        explanations = []
        for factor in top_factors:
            feat_name = factor["feature"]
            raw_val = factor["raw_value"]
            impact = factor["impact_score"]
            
            # Match human-readable template
            base_name = feat_name.replace("tld_", "")
            template = FEATURE_DESCRIPTIONS.get(feat_name, None)
            if template:
                try:
                    desc = template.format(val=raw_val)
                except Exception:
                    desc = f"{feat_name.replace('_', ' ').capitalize()}: {raw_val}"
            elif feat_name.startswith("tld_") and raw_val == 1:
                desc = f"Top-Level Domain '{base_name}' detected in URL."
            else:
                desc = f"{feat_name.replace('_', ' ').capitalize()} recorded value of {raw_val}."

            explanations.append({
                "feature": feat_name,
                "description": desc,
                "raw_value": raw_val,
                "impact_score": round(impact, 4),
                "direction": factor["direction"]
            })

        return {
            "explainability_method": method,
            "top_factors": explanations,
            "all_attributions": top_factors
        }

    def _compute_importance_attribution(self, X_vec: np.ndarray, raw_features: Dict[str, Any], feature_names: List[str]) -> List[Dict[str, Any]]:
        """Fallback attribution using model feature importances multiplied by normalized feature vector."""
        factors = []
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_[0])
        else:
            importances = np.ones(len(feature_names)) / len(feature_names)

        vec = X_vec[0]
        for idx, name in enumerate(feature_names):
            imp = importances[idx] if idx < len(importances) else 0.0
            val_norm = vec[idx] if idx < len(vec) else 0.0
            raw_val = raw_features.get(name, 0)
            score = float(imp * val_norm)
            factors.append({
                "feature": name,
                "raw_value": raw_val,
                "impact_score": score,
                "abs_impact": abs(score),
                "direction": "increases_risk" if score > 0 else "decreases_risk"
            })
        return factors

if __name__ == "__main__":
    expl = ModelExplainer()
    test_url = "http://paypal.account-verify.xyz/login.php?user=admin&token=9928"
    res = expl.explain_url(test_url)
    print(f"Explainability Method: {res['explainability_method']}")
    print("\nTop Contributing Factors:")
    for f in res['top_factors']:
        print(f" - [{f['direction'].upper()}] {f['feature']}: {f['description']} (impact: {f['impact_score']})")
