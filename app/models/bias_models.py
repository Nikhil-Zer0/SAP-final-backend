from pydantic import BaseModel
from typing import Dict, List, Optional, Union, Literal
from enum import Enum

class SensitiveAttribute(str, Enum):
    GENDER = "gender"
    AGE = "age"
    ETHNICITY = "ethnicity"
    RACE = "race"
    DISABILITY = "disability"

class BiasMetric(BaseModel):
    name: str
    value: float
    description: str
    threshold: float
    compliant: bool

class BiasDetectionResponse(BaseModel):
    model_name: str
    model_version: str
    dataset_name: str
    sensitive_attribute: str
    privileged_group: str
    unprivileged_group: str
    metrics: List[BiasMetric]
    audit_status: str
    recommendations: List[str]
    detection_timestamp: str
    record_count: int

    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "hiring_model_v1",
                "model_version": "1.0",
                "dataset_name": "hiring_data.csv",
                "sensitive_attribute": "gender",
                "privileged_group": "Male",
                "unprivileged_group": "Female",
                "audit_status": "NON-COMPLIANT",
                "detection_timestamp": "2025-01-15T10:30:00.000Z",
                "record_count": 100,
                "metrics": [
                    {
                        "name": "Disparate Impact Ratio",
                        "value": 0.429,
                        "description": "Ratio of favorable outcomes between privileged and unprivileged groups (gender)",
                        "threshold": 0.8,
                        "compliant": False
                    }
                ],
                "recommendations": [
                    "Bias detected: Disparate impact ratio below 0.8 (EEOC 80% rule)"
                ]
            }
        }