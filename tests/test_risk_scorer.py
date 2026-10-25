"""
Unit tests for Risk Scoring and Recommendation Engine.
"""

import pytest
from ml.risk_scorer import calculate_risk_assessment

def test_risk_scoring_low():
    res = calculate_risk_assessment(0.05)
    assert res["riskScore"] == 5
    assert res["riskLevel"] == "LOW"
    assert "low-risk" in res["badge"]
    assert "Low-risk" in res["recommendation"]

def test_risk_scoring_medium():
    res = calculate_risk_assessment(0.52)
    assert res["riskScore"] == 52
    assert res["riskLevel"] == "MEDIUM"
    assert "medium-risk" in res["badge"]
    assert "caution" in res["recommendation"].lower()

def test_risk_scoring_high():
    res = calculate_risk_assessment(0.95)
    assert res["riskScore"] == 95
    assert res["riskLevel"] == "HIGH"
    assert "high-risk" in res["badge"]
    assert "High-risk" in res["recommendation"]

def test_risk_scoring_boundaries():
    # Exactly at boundary 0.40 -> MEDIUM
    res_40 = calculate_risk_assessment(0.40)
    assert res_40["riskLevel"] == "MEDIUM"
    
    # Exactly at boundary 0.70 -> HIGH
    res_70 = calculate_risk_assessment(0.70)
    assert res_70["riskLevel"] == "HIGH"
