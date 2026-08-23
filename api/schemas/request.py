"""
CreditIQ API Request Schemas
"""

from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    applicant_id: int = Field(
        ...,
        description="Home Credit applicant ID (SK_ID_CURR)",
        gt=0,
    )


class ExplanationRequest(BaseModel):
    applicant_id: int = Field(
        ...,
        description="Home Credit applicant ID (SK_ID_CURR)",
        gt=0,
    )