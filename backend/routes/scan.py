"""
Scan API Endpoints for Single and Batch URL Cyber Risk Assessment.
"""

from fastapi import APIRouter, HTTPException, status
from typing import List
from backend.schemas import URLScanRequest, URLScanResponse
from backend.services.scan_service import ScanService

router = APIRouter(prefix="/api", tags=["URL Scanning"])

@router.post("/scan", response_model=URLScanResponse, summary="Analyze a single website URL for cybersecurity risks")
async def scan_single_url(request: URLScanRequest):
    """
    Accepts a website URL, extracts security features, runs ML inference with the trained model,
    calculates explainability factors (SHAP), and generates tailored security recommendations.
    """
    try:
        service = ScanService.get_instance()
        result = service.scan_url(
            raw_url=request.url, 
            include_network_probe=request.include_network_probe
        )
        return result
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Analysis pipeline error: {str(e)}"
        )

@router.post("/batch-scan", response_model=List[URLScanResponse], summary="Scan multiple URLs in batch")
async def scan_batch_urls(requests: List[URLScanRequest]):
    """Analyzes a batch of URLs sequentially."""
    if len(requests) > 20:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch limit exceeded. Maximum 20 URLs per request."
        )
    service = ScanService.get_instance()
    results = []
    for req in requests:
        try:
            res = service.scan_url(req.url, include_network_probe=req.include_network_probe)
            results.append(res)
        except Exception as e:
            # Continue processing others
            pass
    return results
