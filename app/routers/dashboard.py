"""
dashboard.py
Dashboard API router for Ethical AI Auditor
Provides CTO-level summary metrics for Sarah Chen
Alternative to SAP Analytics Cloud dashboards
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List, Any
import logging

# Services
from app.services.dashboard.dashboard_service import DashboardService
from app.database.session import get_db
from app.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/dashboard/summary")
async def get_dashboard_summary(conn = Depends(get_db)) -> Dict[str, Any]:
    """
    Get high-level summary metrics for CTO dashboard.
    
    Returns:
        Key KPIs for executive view:
        - Total models audited
        - Compliance rate
        - Top bias source
        - Risk score
        - Trend
    """
    try:
        logger.info("Generating dashboard summary for CTO view")
        
        summary = DashboardService.get_summary_metrics()
        return summary
        
    except Exception as e:
        logger.error(f"Dashboard summary generation failed: {str(e)}")
        raise HTTPException(500, f"Dashboard generation failed: {str(e)}")

@router.get("/dashboard/model_risk")
async def get_model_risk_breakdown(conn = Depends(get_db)) -> List[Dict[str, Any]]:
    """
    Get detailed model risk breakdown.
    
    Returns:
        List of models with risk scores, status, and bias sources
    """
    try:
        logger.info("Generating model risk breakdown")
        
        risk_data = DashboardService.get_model_risk_breakdown()
        return risk_data
        
    except Exception as e:
        logger.error(f"Model risk breakdown failed: {str(e)}")
        raise HTTPException(500, f"Model risk breakdown failed: {str(e)}")

@router.get("/dashboard/compliance_trend")
async def get_compliance_trend(conn = Depends(get_db)) -> List[Dict[str, Any]]:
    """
    Get weekly compliance trend for charting.
    """
    try:
        logger.info("Generating compliance trend data")
        
        trend_data = DashboardService.get_compliance_trend()
        return trend_data
        
    except Exception as e:
        logger.error(f"Compliance trend generation failed: {str(e)}")
        raise HTTPException(500, f"Compliance trend generation failed: {str(e)}")