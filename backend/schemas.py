"""
Pydantic Schemas for API Request and Response Validation.
"""

from pydantic import BaseModel, Field, HttpUrl, field_validator
from typing import List, Dict, Any, Optional
from datetime import datetime

class URLScanRequest(BaseModel):
    url: str = Field(..., description="The website URL to analyze and scan", min_length=3, max_length=2048)
    include_network_probe: bool = Field(default=False, description="Whether to safely query HTTP headers/redirects (with timeout)")

    @field_validator("url")
    @classmethod
    def validate_url_string(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("URL cannot be empty.")
        if len(v) < 3:
            raise ValueError("URL is too short to be a valid web resource.")
        # Basic sanity check
        if not v.startswith(("http://", "https://", "ftp://")):
            v = "http://" + v
        return v

class ContributingFactor(BaseModel):
    feature: str
    description: str
    raw_value: Any
    impact_score: float
    direction: str

class ThresholdConfig(BaseModel):
    low: int
    high: int

class URLScanResponse(BaseModel):
    id: Optional[int] = None
    url: str
    normalized_url: str
    prediction: str
    is_malicious: bool
    riskLevel: str
    riskScore: int
    modelProbability: float
    confidence: str
    color: str
    badge: str
    keyFactors: List[ContributingFactor]
    extractedFeatures: Dict[str, Any]
    recommendation: str
    actionAdvice: str
    explainabilityMethod: str
    executionTimeMs: float
    timestamp: str
    disclaimer: str

class ScanHistoryItem(BaseModel):
    id: int
    url: str
    prediction: str
    risk_level: str
    risk_score: int
    model_probability: float
    key_factor_summary: str
    timestamp: str

class ScanHistoryListResponse(BaseModel):
    total_scans: int
    low_risk_count: int
    medium_risk_count: int
    high_risk_count: int
    page: int
    page_size: int
    items: List[ScanHistoryItem]

class ModelInfoResponse(BaseModel):
    model_name: str
    algorithm: str
    created_at: str
    total_training_samples: int
    total_validation_samples: int
    total_test_samples: int
    feature_count: int
    feature_names: List[str]
    explainer_type: str
    metrics: Dict[str, Any]
    all_models_comparison: Dict[str, Any]
    top_feature_importances: List[Dict[str, Any]]
