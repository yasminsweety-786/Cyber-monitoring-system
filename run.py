"""
Unified Application Launcher for ML-Based Intelligent Cyber Monitoring System.
Starts the FastAPI server hosting both the REST API and the Frontend Dashboard.
"""

import sys
import os
import uvicorn

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def main():
    import socket
    port = 8000
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex(('127.0.0.1', 8000)) == 0:
                port = 8001
    except Exception:
        pass

    print("=" * 65)
    print("  ML-Based Intelligent Cyber Monitoring System")
    print("=" * 65)
    print(f"  Dashboard UI          : http://127.0.0.1:{port}/")
    print(f"  API Docs (Swagger UI) : http://127.0.0.1:{port}/docs")
    print(f"  Health Endpoint       : http://127.0.0.1:{port}/api/health")
    print("=" * 65)
    
    uvicorn.run(
        "backend.main:app", 
        host="127.0.0.1", 
        port=port, 
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()
