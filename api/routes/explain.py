"""
CreditIQ Explain Route

Returns SHAP explanations for one applicant.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.engine import Engine

import pandas as pd
import shap

from api.dependencies import (
    get_database_engine,
    get_model,
    get_calibrator,
    get_threshold,
    get_model_version,
)

from api.schemas.request import ExplanationRequest
from api.schemas.response import (
    ExplanationResponse,
    RiskFactor,
)

router = APIRouter(
    prefix="/explain",
    tags=["Explainability"],
)


def load_features(applicant_id: int, engine: Engine, model):

    query = text(
        """
        SELECT *
        FROM applicant_features
        WHERE sk_id_curr = :id
        """
    )

    with engine.connect() as conn:
        df = pd.read_sql(
            query,
            conn,
            params={"id": applicant_id},
        )

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail="Applicant not found.",
        )

    X = df[list(model.feature_names_in_)].astype(float)

    return X


@router.post(
    "",
    response_model=ExplanationResponse,
)
def explain_applicant(
    request: ExplanationRequest,
    model=Depends(get_model),
    calibrator=Depends(get_calibrator),
    threshold=Depends(get_threshold),
    engine=Depends(get_database_engine),
    model_version=Depends(get_model_version),
):

    X = load_features(
        request.applicant_id,
        engine,
        model,
    )

    raw_prob = float(
        model.predict_proba(X)[0, 1]
    )

    probability = float(
        calibrator.predict_proba(
            [[raw_prob]]
        )[0, 1]
    )

    decision = (
        "DECLINE"
        if probability >= threshold
        else "APPROVE"
    )

    explainer = shap.TreeExplainer(model)

    values = explainer.shap_values(X)

    if isinstance(values, list):
        values = values[0]

    values = values[0]

    factors = pd.DataFrame(
        {
            "feature": model.feature_names_in_,
            "contribution": values,
        }
    )

    risk = (
        factors[factors.contribution > 0]
        .sort_values(
            "contribution",
            ascending=False,
        )
        .head(5)
    )

    protective = (
        factors[factors.contribution < 0]
        .sort_values("contribution")
        .head(5)
    )

    return ExplanationResponse(
        applicant_id=request.applicant_id,
        probability_of_default=probability,
        decision=decision,
        threshold=threshold,
        model_version=model_version,
        top_risk_factors=[
            RiskFactor(
                feature=r.feature,
                contribution=float(r.contribution),
                direction="risk_increasing",
            )
            for _, r in risk.iterrows()
        ],
        top_protective_factors=[
            RiskFactor(
                feature=r.feature,
                contribution=float(r.contribution),
                direction="risk_reducing",
            )
            for _, r in protective.iterrows()
        ],
    )