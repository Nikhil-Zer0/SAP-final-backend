from pydantic import BaseModel
from typing import Dict, List, Any, Optional

class ExplanationRequest(BaseModel):
    model_name: str
    model_version: str
    dataset: List[Dict[str, Any]]
    target_variable: str
    sensitive_attribute: str
    instance_index: int
    role: str = "technical"  # technical, compliance, executive

class FeatureImportance(BaseModel):
    feature: str
    importance: float
    direction: str  # positive, negative

class ExplanationResponse(BaseModel):
    model_name: str
    model_version: str
    instance_index: int
    shap_values: Dict[str, float]
    feature_importance: List[FeatureImportance]
    natural_language_explanation: str
    role: str
    recommendations: List[str]