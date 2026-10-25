"""
Scan History and Statistics API Endpoints.
"""

from fastapi import APIRouter, HTTPException, Query, status
from typing import Optional
from backend.schemas import ScanHistoryListResponse
from backend.database import get_scan_history, get_scan_statistics, clear_scan_history, delete_scan_record

router = APIRouter(prefix="/api/history", tags=["Scan History"])

@router.get("", response_model=ScanHistoryListResponse, summary="Get stored scan history with filtering and pagination")
async def list_scan_history(
    search: Optional[str] = Query(None, description="Search query matching URL or key factors"),
    risk_level: Optional[str] = Query(None, description="Filter by risk tier (LOW, MEDIUM, HIGH, or ALL)"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page")
):
    offset = (page - 1) * page_size
    items, total_matching = get_scan_history(
        search=search, 
        risk_level=risk_level, 
        limit=page_size, 
        offset=offset
    )
    stats = get_scan_statistics()
    
    return {
        "total_scans": total_matching,
        "low_risk_count": stats["low_risk_count"],
        "medium_risk_count": stats["medium_risk_count"],
        "high_risk_count": stats["high_risk_count"],
        "page": page,
        "page_size": page_size,
        "items": items
    }

@router.get("/stats", summary="Get overall dashboard scan statistics")
async def get_stats():
    return get_scan_statistics()

@router.delete("", summary="Clear all scan history")
async def delete_all_history():
    deleted = clear_scan_history()
    return {"message": f"Successfully cleared {deleted} scan records.", "deleted_count": deleted}

@router.delete("/{record_id}", summary="Delete a specific scan record by ID")
async def delete_single_history(record_id: int):
    success = delete_scan_record(record_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scan record not found.")
    return {"message": f"Scan record {record_id} deleted successfully."}
