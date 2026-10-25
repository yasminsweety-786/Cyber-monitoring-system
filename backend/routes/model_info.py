import os
import json
import math
from fastapi import APIRouter, HTTPException, status
from backend.config import METADATA_FILE, EVALUATION_FILE, DATASET_INFO_FILE

router = APIRouter(prefix="/api", tags=["Model & Dataset Intelligence"])

def sanitize_json_val(val):
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    elif isinstance(val, dict):
        return {k: sanitize_json_val(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [sanitize_json_val(v) for v in val]
    return val

@router.get("/model-info", summary="Retrieve trained model architecture, metrics, and candidate comparison")
async def get_model_info():
    if not os.path.exists(METADATA_FILE) or not os.path.exists(EVALUATION_FILE):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Model metadata not found. The ML pipeline has not been executed yet."
        )
        
    with open(METADATA_FILE, "r") as f:
        meta = json.load(f)
        
    with open(EVALUATION_FILE, "r") as f:
        eval_data = json.load(f)
        
    response = {
        "model_name": meta.get("model_name"),
        "algorithm": meta.get("algorithm"),
        "created_at": meta.get("created_at"),
        "total_training_samples": meta.get("total_training_samples"),
        "total_validation_samples": meta.get("total_validation_samples"),
        "total_test_samples": meta.get("total_test_samples"),
        "feature_count": meta.get("feature_count"),
        "feature_names": meta.get("feature_names", []),
        "explainer_type": meta.get("explainer_type"),
        "metrics": meta.get("metrics", {}),
        "hyperparameters": meta.get("hyperparameters", {}),
        "all_models_comparison": eval_data.get("models", {}),
        "top_feature_importances": eval_data.get("top_features", [])
    }
    return sanitize_json_val(response)

@router.get("/dataset-info", summary="Retrieve dataset provenance, balance, and exploratory statistics")
async def get_dataset_info():
    if not os.path.exists(DATASET_INFO_FILE):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Dataset information not found. Run dataset inspection first."
        )
    with open(DATASET_INFO_FILE, "r") as f:
        return json.load(f)

@router.get("/features-list", summary="Retrieve the engineered feature registry and descriptions")
async def get_features_list():
    from ml.explainer import FEATURE_DESCRIPTIONS
    from ml.preprocessor import NUMERICAL_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES
    
    return {
        "total_engineered_features": len(NUMERICAL_FEATURES) + len(BINARY_FEATURES) + len(CATEGORICAL_FEATURES),
        "numerical_features": NUMERICAL_FEATURES,
        "binary_features": BINARY_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_descriptions": FEATURE_DESCRIPTIONS
    }
