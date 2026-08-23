"""
CreditIQ — Fairness Audit Tests
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation.parquet"
)

MODEL_VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation_final.parquet"
)

FAIRNESS_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "metrics"
    / "fairness_audit.json"
)


# ============================================================
# Test age proxy source
# ============================================================

def test_days_birth_exists_in_raw_validation():

    validation = pd.read_parquet(
        RAW_VALIDATION_FILE
    )

    assert "DAYS_BIRTH" in validation.columns


# ============================================================
# Test model validation dataset
# ============================================================

def test_model_validation_exists():

    validation = pd.read_parquet(
        MODEL_VALIDATION_FILE
    )

    assert len(validation) > 0
    assert "TARGET" in validation.columns


# ============================================================
# Test validation alignment
# ============================================================

def test_validation_row_alignment():

    raw_validation = pd.read_parquet(
        RAW_VALIDATION_FILE
    )

    model_validation = pd.read_parquet(
        MODEL_VALIDATION_FILE
    )

    assert len(raw_validation) == len(
        model_validation
    )

    assert np.array_equal(
        raw_validation["TARGET"].to_numpy(),
        model_validation["TARGET"].to_numpy(),
    )


# ============================================================
# Test age groups
# ============================================================

def test_multiple_age_groups_exist():

    validation = pd.read_parquet(
        RAW_VALIDATION_FILE
    )

    age_years = (
        validation["DAYS_BIRTH"].abs()
        / 365.25
    )

    age_groups = pd.cut(
        age_years,
        bins=[
            0,
            25,
            35,
            45,
            55,
            np.inf,
        ],
        labels=[
            "<25",
            "25-34",
            "35-44",
            "45-54",
            "55+",
        ],
        right=False,
    )

    assert age_groups.nunique() >= 2


# ============================================================
# Test fairness report
# ============================================================

def test_fairness_report_exists():

    assert FAIRNESS_REPORT.exists()


# ============================================================
# Test fairness report structure
# ============================================================

def test_fairness_report_structure():

    with open(
        FAIRNESS_REPORT,
        "r",
        encoding="utf-8",
    ) as file:

        report = json.load(file)

    assert (
        "before_mitigation"
        in report
    )

    assert (
        "after_mitigation"
        in report
    )

    assert (
        "primary_fairness_metric"
        in report
    )

    assert (
        report["primary_fairness_metric"]
        == "equalized_odds_difference"
    )


# ============================================================
# Test fairness metrics
# ============================================================

def test_fairness_metrics_are_finite():

    with open(
        FAIRNESS_REPORT,
        "r",
        encoding="utf-8",
    ) as file:

        report = json.load(file)

    before = report[
        "before_mitigation"
    ]["fairness"]

    after = report[
        "after_mitigation"
    ]["fairness"]

    for metrics in [
        before,
        after,
    ]:

        assert np.isfinite(
            metrics[
                "demographic_parity_difference"
            ]
        )

        assert np.isfinite(
            metrics[
                "equalized_odds_difference"
            ]
        )


# ============================================================
# Test mitigation reduced equalized odds disparity
# ============================================================

def test_equalized_odds_improved():

    with open(
        FAIRNESS_REPORT,
        "r",
        encoding="utf-8",
    ) as file:

        report = json.load(file)

    before = abs(
        report[
            "before_mitigation"
        ]["fairness"][
            "equalized_odds_difference"
        ]
    )

    after = abs(
        report[
            "after_mitigation"
        ]["fairness"][
            "equalized_odds_difference"
        ]
    )

    assert after < before


# ============================================================
# Test frozen threshold
# ============================================================

def test_frozen_threshold():

    with open(
        FAIRNESS_REPORT,
        "r",
        encoding="utf-8",
    ) as file:

        report = json.load(file)

    assert (
        report["decision_threshold"]
        == 0.20
    )


# ============================================================
# Test test-set isolation
# ============================================================

def test_fairness_audit_does_not_use_test_set():

    with open(
        FAIRNESS_REPORT,
        "r",
        encoding="utf-8",
    ) as file:

        report = json.load(file)

    assert (
        report["test_set_used"]
        is False
    )