"""
CreditIQ Score Route

Endpoint:
    POST /score

Flow:
    Applicant ID
        ↓
    PostgreSQL Feature Store
        ↓
    80 model features
        ↓
    Monotonic XGBoost
        ↓
    Platt calibration
        ↓
    Frozen threshold
        ↓
    APPROVE / DECLINE
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
import json
from sqlalchemy.engine import Engine

import pandas as pd

from api.dependencies import (
    get_database_engine,
    get_model,
    get_calibrator,
    get_threshold,
    get_model_version,
)

from api.schemas.request import ScoreRequest
from api.schemas.response import ScoreResponse


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/score",
    tags=["Scoring"],
)


# ============================================================
# LOAD APPLICANT FEATURES
# ============================================================

def load_applicant_features(
    applicant_id: int,
    engine: Engine,
    model,
) -> pd.DataFrame:
    """
    Retrieve one applicant from the PostgreSQL
    model-ready feature store.
    """

    query = text(
        """
        SELECT *
        FROM applicant_features
        WHERE sk_id_curr = :applicant_id
        """
    )

    with engine.connect() as connection:

        df = pd.read_sql(
            query,
            connection,
            params={
                "applicant_id": applicant_id,
            },
        )

    if df.empty:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Applicant {applicant_id} "
                "was not found in the feature store."
            ),
        )

    # --------------------------------------------------------
    # Verify all model features exist
    # --------------------------------------------------------

    model_features = list(
        model.feature_names_in_
    )

    missing_features = [
        feature
        for feature in model_features
        if feature not in df.columns
    ]

    if missing_features:

        raise HTTPException(
            status_code=500,
            detail=(
                "Feature store is missing model features: "
                + ", ".join(missing_features)
            ),
        )

    # --------------------------------------------------------
    # Create model matrix
    # --------------------------------------------------------

    X = df[
        model_features
    ].copy()

    # PostgreSQL feature store should already contain
    # numeric WOE-transformed features.
    try:

        X = X.astype(float)

    except (TypeError, ValueError) as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Model features could not be converted "
                f"to numeric values: {error}"
            ),
        )

    # --------------------------------------------------------
    # Missing-value validation
    # --------------------------------------------------------

    if X.isna().any().any():

        missing = X.columns[
            X.isna().any()
        ].tolist()

        raise HTTPException(
            status_code=500,
            detail=(
                "Missing values detected in model features: "
                + ", ".join(missing)
            ),
        )

    return X
# ============================================================
# AUDIT LOGGING
# ============================================================

def write_scoring_audit(
    applicant_id: int,
    request_payload: dict,
    predicted_probability: float,
    decision: str,
    model_version: str,
    engine: Engine,
):
    """
    Store every scoring decision for traceability and audit.
    """

    # Simple decision-level reason code.
    # Detailed SHAP explanations are provided by /explain.
    if decision == "DECLINE":
        reason_codes = [
            "probability_above_decision_threshold"
        ]
    else:
        reason_codes = [
            "probability_below_decision_threshold"
        ]

    query = text(
        """
        INSERT INTO scoring_audit_log (
            sk_id_curr,
            request_payload,
            predicted_probability,
            decision,
            top_reason_codes,
            model_version
        )
        VALUES (
            :sk_id_curr,
            CAST(:request_payload AS JSONB),
            :predicted_probability,
            :decision,
            CAST(:top_reason_codes AS JSONB),
            :model_version
        )
        """
    )

    with engine.begin() as connection:

        connection.execute(
            query,
            {
                "sk_id_curr": applicant_id,
                "request_payload": json.dumps(
                    request_payload
                ),
                "predicted_probability": predicted_probability,
                "decision": decision,
                "top_reason_codes": json.dumps(
                    reason_codes
                ),
                "model_version": model_version,
            },
        )

# ============================================================
# SCORE APPLICANT
# ============================================================

@router.post(
    "",
    response_model=ScoreResponse,
)
def score_applicant(
    request: ScoreRequest,
    model=Depends(get_model),
    calibrator=Depends(get_calibrator),
    threshold: float = Depends(get_threshold),
    engine: Engine = Depends(get_database_engine),
    model_version: str = Depends(get_model_version),
):
    """
    Score a single applicant.

    Returns:
    - calibrated probability of default
    - credit decision
    - frozen decision threshold
    - model version
    """

    # --------------------------------------------------------
    # Load applicant
    # --------------------------------------------------------

    X = load_applicant_features(
        applicant_id=request.applicant_id,
        engine=engine,
        model=model,
    )

    # --------------------------------------------------------
    # Raw XGBoost probability
    # --------------------------------------------------------

    raw_probability = float(
        model.predict_proba(X)[0, 1]
    )

    # --------------------------------------------------------
    # Platt calibration
    # --------------------------------------------------------

    calibrated_probability = float(
        calibrator.predict_proba(
            [[raw_probability]]
        )[0, 1]
    )

        # --------------------------------------------------------
    # Frozen decision threshold
    # --------------------------------------------------------

    decision = (
        "DECLINE"
        if calibrated_probability >= threshold
        else "APPROVE"
    )

    # --------------------------------------------------------
    # Write scoring decision to audit log
    # --------------------------------------------------------

    write_scoring_audit(
        applicant_id=request.applicant_id,
        request_payload={
            "applicant_id": request.applicant_id
        },
        predicted_probability=calibrated_probability,
        decision=decision,
        model_version=model_version,
        engine=engine,
    )

    return ScoreResponse(
        applicant_id=request.applicant_id,
        probability_of_default=calibrated_probability,
        decision=decision,
        threshold=threshold,
        model_version=model_version,
    )