"""
CreditIQ API Dependencies

Centralized loading of:
- Final XGBoost model
- Platt calibrator
- Frozen decision threshold
- PostgreSQL connection
"""

from functools import lru_cache
from pathlib import Path

import joblib
import yaml

from src.data.db_loader import create_database_engine


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost_monotonic.joblib"
)

CALIBRATOR_PATH = (
    PROJECT_ROOT
    / "models"
    / "platt_calibrator.joblib"
)

THRESHOLD_PATH = (
    PROJECT_ROOT
    / "configs"
    / "threshold.yaml"
)


# ============================================================
# MODEL
# ============================================================

@lru_cache(maxsize=1)
def get_model():
    """
    Load the finalized monotonic XGBoost model.

    The model is cached so it is loaded only once
    during the API process lifetime.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


# ============================================================
# CALIBRATOR
# ============================================================

@lru_cache(maxsize=1)
def get_calibrator():
    """
    Load the Platt calibration model.
    """

    if not CALIBRATOR_PATH.exists():
        raise FileNotFoundError(
            f"Calibrator not found:\n{CALIBRATOR_PATH}"
        )

    return joblib.load(CALIBRATOR_PATH)


# ============================================================
# DECISION THRESHOLD
# ============================================================

@lru_cache(maxsize=1)
def get_threshold() -> float:
    """
    Load the frozen decision threshold.

    The threshold was selected using validation data
    and must not be changed during API scoring.
    """

    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Threshold configuration not found:\n"
            f"{THRESHOLD_PATH}"
        )

    with open(
        THRESHOLD_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        config = yaml.safe_load(file)

    threshold = config.get(
        "decision_threshold"
    )

    if threshold is None:
        raise ValueError(
            "decision_threshold is missing "
            "from threshold.yaml"
        )

    return float(threshold)


# ============================================================
# DATABASE
# ============================================================

@lru_cache(maxsize=1)
def get_database_engine():
    """
    Create and cache the PostgreSQL database engine.
    """

    return create_database_engine()


# ============================================================
# MODEL METADATA
# ============================================================

def get_model_version() -> str:
    """
    Return the current CreditIQ model version.
    """

    return "xgboost_monotonic_v1"


# ============================================================
# HEALTH CHECK HELPERS
# ============================================================

def check_model_loaded() -> bool:
    """
    Verify that the model can be loaded.
    """

    try:
        get_model()
        return True

    except Exception:
        return False


def check_database_connection() -> bool:
    """
    Verify PostgreSQL connectivity.
    """

    try:
        engine = get_database_engine()

        with engine.connect():
            pass

        return True

    except Exception:
        return False