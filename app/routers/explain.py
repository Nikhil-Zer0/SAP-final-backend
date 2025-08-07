"""
explain.py
Optimized Explainability API router - Fixed for SHAP numeric issues
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from typing import Dict, Any
import logging

import pandas as pd

# Services
from app.services.explainability.shap_service import SHAPService
from app.services.explainability.llm_service import LLMService
from app.database.session import get_db
from app.config import settings
from app.utils.data_processor import process_csv_file

# Response models
from pydantic import BaseModel
from typing import List as TypingList

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/explain/debug")
async def debug_explain_cache():
    """Debug endpoint to check cache status and clear if needed."""
    from app.services.explainability.shap_service import SHAPService
    
    cache_info = {
        "model_cache_keys": list(SHAPService._model_cache.keys()),
        "data_cache_keys": list(SHAPService._data_cache.keys()),
        "encoder_cache_keys": list(SHAPService._encoder_cache.keys())
    }
    
    return cache_info

@router.delete("/explain/cache")
async def clear_explain_cache():
    """Clear all explanation caches for testing."""
    from app.services.explainability.shap_service import SHAPService
    
    SHAPService._model_cache.clear()
    SHAPService._data_cache.clear()
    SHAPService._encoder_cache.clear()
    
    return {"message": "All explanation caches cleared"}

@router.get("/explain/accuracy")
async def get_model_accuracy_stats():
    """Get accuracy statistics for cached models."""
    from app.services.explainability.shap_service import SHAPService
    from sklearn.model_selection import train_test_split
    
    accuracy_stats = {}
    
    for cache_key, model in SHAPService._model_cache.items():
        try:
            # Get cached data
            if cache_key in SHAPService._data_cache:
                X = SHAPService._data_cache[cache_key]['X']
                y = SHAPService._data_cache[cache_key]['y']
                
                # Calculate train/test split accuracy
                if len(X) > 10:  # Only if we have enough data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.3, random_state=42, stratify=y
                    )
                    
                    model_clone = model.__class__(**model.get_params())
                    model_clone.fit(X_train, y_train)
                    
                    train_acc = model_clone.score(X_train, y_train)
                    test_acc = model_clone.score(X_test, y_test)
                    
                    accuracy_stats[cache_key] = {
                        "train_accuracy": round(train_acc, 3),
                        "test_accuracy": round(test_acc, 3),
                        "overfitting_gap": round(train_acc - test_acc, 3),
                        "dataset_size": len(X),
                        "feature_count": len(X.columns),
                        "class_distribution": {
                            "class_0": int((y == 0).sum()),
                            "class_1": int((y == 1).sum())
                        }
                    }
                else:
                    accuracy_stats[cache_key] = {
                        "error": "Insufficient data for accuracy calculation",
                        "dataset_size": len(X)
                    }
        except Exception as e:
            accuracy_stats[cache_key] = {
                "error": f"Could not calculate accuracy: {str(e)}"
            }
    
    return {
        "cached_models": len(SHAPService._model_cache),
        "accuracy_stats": accuracy_stats,
        "timestamp": pd.Timestamp.now().isoformat()
    }

class FeatureImportanceResponse(BaseModel):
    feature: str
    importance: float
    direction: str

class ExplanationResponse(BaseModel):
    model_name: str
    model_version: str
    instance_index: int
    shap_values: Dict[str, float]
    feature_importance: TypingList[FeatureImportanceResponse]
    natural_language_explanation: str
    role: str
    recommendations: TypingList[str]

@router.post("/explain", response_model=ExplanationResponse)
async def explain_model(
    model_name: str = Form(...),
    model_version: str = Form(...),
    target_variable: str = Form(...),
    sensitive_attribute: str = Form(...),
    instance_index: int = Form(...),
    role: str = Form("technical"),
    file: UploadFile = File(...),
    conn = Depends(get_db)
):
    try:
        logger.info(f"Received explanation request for model: {model_name}")
        
        # Validate file
        if file.content_type not in ["text/csv", "application/vnd.ms-excel"]:
            raise HTTPException(400, "Invalid file format. Please upload a CSV file.")
        
        # Process CSV
        file_content = await file.read()
        dataset = process_csv_file(file_content)
        
        # Validate index
        if instance_index >= len(dataset):
            raise HTTPException(400, f"instance_index {instance_index} out of range")
        
        # Train or get cached model
        try:
            model, X, y = SHAPService.train_model(dataset, target_variable)
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise HTTPException(500, f"Could not train model: {str(e)}")
        
        # Get SHAP values
        try:
            shap_values = SHAPService.get_shap_values(model, X, instance_index)
        except Exception as e:
            logger.error(f"SHAP computation failed: {str(e)}")
            raise HTTPException(500, f"SHAP computation failed: {str(e)}")
        
        # Get feature importance
        feature_importance_objs = SHAPService.get_feature_importance(shap_values)
        feature_importance = [
            {"feature": f.feature, "importance": f.importance, "direction": f.direction}
            for f in feature_importance_objs
        ]
        
        # Generate explanations
        try:
            explanation = LLMService.generate_explanation(
                shap_values, feature_importance_objs,
                target_variable, sensitive_attribute, role
            )
            recommendations = LLMService.generate_recommendations(shap_values, sensitive_attribute)
        except Exception as e:
            logger.error(f"LLM service failed: {str(e)}")
            raise HTTPException(500, f"Explanation generation failed: {str(e)}")
        
        return ExplanationResponse(
            model_name=model_name,
            model_version=model_version,
            instance_index=instance_index,
            shap_values=shap_values,
            feature_importance=feature_importance,
            natural_language_explanation=explanation,
            role=role,
            recommendations=recommendations
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(500, f"Internal server error: {str(e)}")