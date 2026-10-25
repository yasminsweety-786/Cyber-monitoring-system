"""
Unit tests for Model Inference and SHAP Explainability.
"""

import os
import pytest
from ml.explainer import ModelExplainer
from backend.services.scan_service import ScanService

def test_model_and_explainer_execution():
    service = ScanService.get_instance()
    
    # Benign URL scan
    benign_url = "https://www.wikipedia.org/wiki/Main_Page"
    res_benign = service.scan_url(benign_url)
    
    assert res_benign["url"] == benign_url
    assert res_benign["riskLevel"] in ["LOW", "MEDIUM"]
    assert "keyFactors" in res_benign
    assert len(res_benign["keyFactors"]) > 0
    assert "extractedFeatures" in res_benign
    
    # Phishing-like deceptive URL scan
    phish_url = "http://paypal.com.account-update.xyz/login.php?cmd=_login-run&token=8fa9b2"
    res_phish = service.scan_url(phish_url)
    
    assert res_phish["riskLevel"] in ["MEDIUM", "HIGH"]
    assert res_phish["riskScore"] > res_benign["riskScore"]
    assert len(res_phish["keyFactors"]) > 0

def test_explainer_local_attribution():
    expl = ModelExplainer()
    res = expl.explain_url("http://secure-banking-alert.xyz/verify")
    
    assert "top_factors" in res
    assert len(res["top_factors"]) > 0
    for factor in res["top_factors"]:
        assert "feature" in factor
        assert "description" in factor
        assert "impact_score" in factor
