"""
compliance.py
Compliance API router for Ethical AI Auditor
Generates audit-ready reports by reusing /explain endpoint
No duplicate SHAP or LLM calls — clean, efficient, SAP-native
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from typing import Dict, Any
import logging
import httpx  # For internal API calls

# Services
from app.services.compliance.report_generator import ComplianceReportGenerator
from app.services.bias_service import BiasDetectionService
from app.utils.data_processor import process_csv_file
from app.database.session import get_db
from app.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/compliance/generate", response_class=FileResponse)
async def generate_compliance_report(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    model_version: str = Form(...),
    target_variable: str = Form(...),
    sensitive_attribute: str = Form(...),
    privileged_group: str = Form(...),
    unprivileged_group: str = Form(...),
    instance_index: int = Form(0),
    role: str = Form("executive"),
    conn = Depends(get_db)
):
    """
    Generate a full compliance report by reusing existing microservices:
    1. Detect bias (FairML)
    2. Call /explain for SHAP + LLM explanation
    3. Generate PDF report
    
    Avoids duplicate model training or LLM calls.
    """
    try:
        logger.info(f"Generating compliance report for model: {model_name}")
        
        # Step 1: Read and process file
        file_content = await file.read()
        dataset = process_csv_file(file_content)
        
        # Step 2: Detect Bias
        try:
            metrics, audit_status, recommendations, count = BiasDetectionService.detect_bias(
                dataset=dataset,
                target_variable=target_variable,
                sensitive_attribute=sensitive_attribute,
                privileged_group=privileged_group,
                unprivileged_group=unprivileged_group
            )
        except Exception as e:
            logger.error(f"Bias detection failed: {str(e)}")
            raise HTTPException(400, f"Bias detection failed: {str(e)}")
        
        # Step 3: Reuse /explain endpoint (Microservice Call)
        try:
            # Simulate the /explain request
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:8000/api/v1/explain",
                    data={
                        "model_name": model_name,
                        "model_version": model_version,
                        "target_variable": target_variable,
                        "sensitive_attribute": sensitive_attribute,
                        "instance_index": instance_index,
                        "role": role
                    },
                    files={"file": ("dataset.csv", file_content, "text/csv")}
                )
                
                if response.status_code != 200:
                    logger.warning(f"/explain call failed: {response.status_code} - {response.text}")
                    shap_explanation = (
                        f"Explainability analysis could not be completed. "
                        f"Model shows potential bias based on {sensitive_attribute}."
                    )
                else:
                    explain_data = response.json()
                    shap_explanation = explain_data.get("natural_language_explanation", "No explanation available.")
                    
        except Exception as e:
            logger.error(f"Failed to call /explain: {str(e)}")
            shap_explanation = (
                f"Explainability analysis temporarily unavailable. "
                f"Model shows potential bias based on {sensitive_attribute}."
            )

        # Step 4: Generate PDF
        report_path = "compliance_report.pdf"
        ComplianceReportGenerator.generate_pdf_report(
            model_name=model_name,
            model_version=model_version,
            bias_metrics=metrics,
            shap_explanation=shap_explanation,
            recommendations=recommendations,
            output_path=report_path
        )
        
        logger.info(f"Compliance report generated: {report_path}")
        
        # Return the PDF file
        return FileResponse(
            path=report_path,
            filename="Ethical_AI_Compliance_Report.pdf",
            media_type="application/pdf"
        )
        
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        raise HTTPException(500, f"Report generation failed: {str(e)}")