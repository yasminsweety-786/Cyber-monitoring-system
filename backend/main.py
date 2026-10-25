"""
Main FastAPI Application Entrypoint for Cyber Monitoring System.
Hosts REST API routes and serves the cybersecurity dashboard frontend.
"""

import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.routes.scan import router as scan_router
from backend.routes.history import router as history_router
from backend.routes.model_info import router as model_info_router
from backend.routes.health import router as health_router

app = FastAPI(
    title="ML-Based Intelligent Cyber Monitoring System",
    description="Real-time Machine Learning URL Risk Assessment, Feature Extraction, SHAP Explainability & Threat Advisory API.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for all local development and cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(health_router)
app.include_router(scan_router)
app.include_router(history_router)
app.include_router(model_info_router)

# Mount Frontend Static Directory if it exists
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        index_file = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"message": "Cyber Monitoring API is active. Open /docs for Swagger UI."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
