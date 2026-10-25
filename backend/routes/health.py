"""
Health and System Status Endpoints.
"""

import os
from datetime import datetime
from fastapi import APIRouter
from backend.config import MODEL_FILE, PREPROCESSOR_FILE

router = APIRouter(prefix="/api", tags=["System Health"])

@router.get("/health", summary="System health and ML model status check")
async def health_check():
    model_loaded = os.path.exists(MODEL_FILE)
    preprocessor_loaded = os.path.exists(PREPROCESSOR_FILE)
    
    return {
        "status": "healthy" if (model_loaded and preprocessor_loaded) else "degraded",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_loaded,
        "preprocessor_loaded": preprocessor_loaded,
        "version": "1.0.0",
        "service": "ML-Based Intelligent Cyber Monitoring System"
    }
