"""
SQLite Database Layer for Scan History and Cyber Monitoring Statistics.
Provides persistent storage, querying, searching, filtering, and metric aggregations.
"""

import os
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from backend.config import DATABASE_PATH

def get_db_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(DATABASE_PATH)), exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the SQLite tables and indexes."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            normalized_url TEXT NOT NULL,
            prediction TEXT NOT NULL,
            is_malicious INTEGER NOT NULL,
            risk_level TEXT NOT NULL,
            risk_score INTEGER NOT NULL,
            model_probability REAL NOT NULL,
            confidence TEXT,
            key_factor_summary TEXT,
            raw_factors_json TEXT,
            extracted_features_json TEXT,
            recommendation TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Indexes for fast query and filtering
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_scan_timestamp ON scan_history(timestamp DESC)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_scan_risk_level ON scan_history(risk_level)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_scan_url ON scan_history(url)")
    
    conn.commit()
    conn.close()

def save_scan_record(scan_data: Dict[str, Any]) -> int:
    """Inserts a completed scan into the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Build a brief key factor summary
    factors = scan_data.get("keyFactors", [])
    factor_strs = [f"{f.get('feature')}" for f in factors[:3]]
    summary = ", ".join(factor_strs) if factor_strs else "Standard features"
    
    cursor.execute("""
        INSERT INTO scan_history (
            url, normalized_url, prediction, is_malicious, 
            risk_level, risk_score, model_probability, confidence,
            key_factor_summary, raw_factors_json, extracted_features_json,
            recommendation, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        scan_data["url"],
        scan_data.get("normalized_url", scan_data["url"]),
        scan_data["prediction"],
        1 if scan_data.get("is_malicious", False) else 0,
        scan_data["riskLevel"],
        scan_data["riskScore"],
        scan_data["modelProbability"],
        scan_data.get("confidence", ""),
        summary,
        json.dumps(scan_data.get("keyFactors", [])),
        json.dumps(scan_data.get("extractedFeatures", {})),
        scan_data.get("recommendation", ""),
        scan_data.get("timestamp", datetime.now().isoformat())
    ))
    
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id

def get_scan_history(
    search: Optional[str] = None,
    risk_level: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
) -> Tuple[List[Dict[str, Any]], int]:
    """Fetches paginated scan history with optional search and filter."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = "SELECT * FROM scan_history WHERE 1=1"
    params = []
    
    if search:
        query += " AND (url LIKE ? OR key_factor_summary LIKE ?)"
        params.extend([f"%{search}%", f"%{search}%"])
        
    if risk_level and risk_level.upper() != "ALL":
        query += " AND risk_level = ?"
        params.append(risk_level.upper())
        
    # Count total matching
    count_query = query.replace("SELECT *", "SELECT COUNT(*)")
    cursor.execute(count_query, params)
    total_count = cursor.fetchone()[0]
    
    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    
    items = []
    for r in rows:
        items.append({
            "id": r["id"],
            "url": r["url"],
            "prediction": r["prediction"],
            "risk_level": r["risk_level"],
            "risk_score": r["risk_score"],
            "model_probability": r["model_probability"],
            "key_factor_summary": r["key_factor_summary"],
            "timestamp": r["timestamp"]
        })
        
    conn.close()
    return items, total_count

def get_scan_statistics() -> Dict[str, int]:
    """Returns total scan count and breakdown by risk category."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM scan_history")
    total = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM scan_history WHERE risk_level = 'LOW'")
    low = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM scan_history WHERE risk_level = 'MEDIUM'")
    medium = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM scan_history WHERE risk_level = 'HIGH'")
    high = cursor.fetchone()[0]
    
    conn.close()
    return {
        "total_scans": total,
        "low_risk_count": low,
        "medium_risk_count": medium,
        "high_risk_count": high
    }

def clear_scan_history() -> int:
    """Deletes all records from scan history."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM scan_history")
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted

def delete_scan_record(record_id: int) -> bool:
    """Deletes a single scan record by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM scan_history WHERE id = ?", (record_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

# Ensure DB table is initialized on import
init_db()
