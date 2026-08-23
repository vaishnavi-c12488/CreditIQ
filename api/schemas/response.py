"""
CreditIQ API Response Schemas
"""

from typing import List

from pydantic import BaseModel


class RiskFactor(BaseModel):
    feature: str
    contribution: float
    direction: str


class ScoreResponse(BaseModel):
    applicant_id: int
    probability_of_default: float
    decision: str
    threshold: float
    model_version: str


class ExplanationResponse(BaseModel):
    applicant_id: int
    probability_of_default: float
    decision: str
    threshold: float
    model_version: str
    top_risk_factors: List[RiskFactor]
    top_protective_factors: List[RiskFactor]


class HealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool
    database_connected: bool