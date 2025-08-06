from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from typing import Dict, Any
from datetime import datetime
import logging

from app.services.bias_service import BiasDetectionService
from app.models.bias_models import BiasDetectionResponse
from app.database.session import get_db
from app.config import settings
from app.utils.data_processor import process_csv_file

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/bias/detect", response_model=BiasDetectionResponse)
async def detect_bias(
    model_name: str = Form(...),
    model_version: str = Form(...),
    target_variable: str = Form(...),
    sensitive_attribute: str = Form(...),
    privileged_group: str = Form(...),
    unprivileged_group: str = Form(...),
    file: UploadFile = File(...),
    conn = Depends(get_db)
):
    """
    Detect bias in an AI model using fairness metrics.
    
    This endpoint analyzes the provided dataset to identify potential biases
    across sensitive attributes like gender, age, or ethnicity.
    
    Returns comprehensive bias metrics and compliance status based on EEOC guidelines.
    """
    try:
        # Log the request
        logger.info(f"Received bias detection request for model: {model_name}")
        
        # Validate file type
        if file.content_type not in ["text/csv", "application/vnd.ms-excel"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Please upload a CSV file."
            )
        
        # Process the CSV file
        file_content = await file.read()
        dataset = process_csv_file(file_content)
        
        # Detect bias using service
        metrics, audit_status, recommendations, record_count = BiasDetectionService.detect_bias(
            dataset=dataset,
            target_variable=target_variable,
            sensitive_attribute=sensitive_attribute,
            privileged_group=privileged_group,
            unprivileged_group=unprivileged_group
        )
        
        # Format metrics
        formatted_metrics = BiasDetectionService.format_metrics(
            metrics, 
            sensitive_attribute
        )
        
        # Generate timestamp
        timestamp = datetime.now().isoformat()
        
        # Save to database
        try:
            conn.execute("""
                INSERT INTO bias_metrics (
                    model_name, model_version, dataset_name, sensitive_attribute,
                    disparate_impact, statistical_parity, equal_opportunity,
                    detection_timestamp, audit_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_name,
                model_version,
                file.filename,
                sensitive_attribute,
                metrics["disparate_impact"],
                metrics["statistical_parity_difference"],
                metrics["equal_opportunity_difference"],
                timestamp,
                audit_status
            ))
            logger.info(f"Saved bias metrics to database for model: {model_name}")
        except Exception as db_err:
            logger.error(f"Database error: {str(db_err)}")
        
        # Return response
        return BiasDetectionResponse(
            model_name=model_name,
            model_version=model_version,
            dataset_name=file.filename,
            sensitive_attribute=sensitive_attribute,
            privileged_group=str(privileged_group),
            unprivileged_group=str(unprivileged_group),
            metrics=formatted_metrics,
            audit_status=audit_status,
            recommendations=recommendations,
            detection_timestamp=timestamp,
            record_count=record_count
        )
        
    except ValueError as ve:
        logger.error(f"Data validation error: {str(ve)}")
        raise HTTPException(
            status_code=400,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"Bias detection failed: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Bias detection failed: {str(e)}"
        )

@router.get("/bias/metrics/{model_name}")
async def get_bias_metrics(
    model_name: str,
    conn = Depends(get_db)
):
    """
    Retrieve historical bias metrics for a specific model.
    
    Returns all bias detection results for the specified model,
    allowing for trend analysis and bias drift monitoring.
    """
    try:
        # Query database for bias metrics
        results = conn.execute("""
            SELECT * FROM bias_metrics 
            WHERE model_name = ?
            ORDER BY detection_timestamp DESC
        """, (model_name,)).fetchall()
        
        # Convert to dictionary format
        columns = [desc[0] for desc in conn.description]
        metrics = [dict(zip(columns, row)) for row in results]
        
        if not metrics:
            raise HTTPException(
                status_code=404,
                detail=f"No bias metrics found for model: {model_name}"
            )
        
        return {
            "model_name": model_name,
            "metrics_count": len(metrics),
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Failed to retrieve bias metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve bias metrics: {str(e)}"
        )