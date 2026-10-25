"""
Integration tests for FastAPI REST Endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

def test_model_info_endpoint():
    response = client.get("/api/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "metrics" in data
    assert "all_models_comparison" in data
    assert "top_feature_importances" in data

def test_dataset_info_endpoint():
    response = client.get("/api/dataset-info")
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] > 0
    assert "class_distribution" in data

def test_scan_api_success():
    payload = {
        "url": "https://www.github.com/torvalds/linux",
        "include_network_probe": False
    }
    response = client.post("/api/scan", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["url"] == payload["url"]
    assert "riskScore" in data
    assert "riskLevel" in data
    assert "keyFactors" in data
    assert "recommendation" in data

def test_scan_api_empty_url():
    payload = {"url": "   "}
    response = client.post("/api/scan", json=payload)
    assert response.status_code == 422 # Validation error

def test_scan_history_workflow():
    # 1. Fetch history
    res_list = client.get("/api/history")
    assert res_list.status_code == 200
    data = res_list.json()
    assert "total_scans" in data
    assert "items" in data
    
    # 2. Fetch stats
    res_stats = client.get("/api/history/stats")
    assert res_stats.status_code == 200
