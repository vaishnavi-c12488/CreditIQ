"""
CreditIQ WOE Encoder Tests
"""

import numpy as np
import pandas as pd
import pytest

from src.features.woe_encoder import calculate_woe_iv


# ============================================================
# NUMERIC FEATURE
# ============================================================

def test_numeric_feature_woe_iv():

    df = pd.DataFrame(
        {
            "income": [
                10000,
                15000,
                20000,
                25000,
                30000,
                35000,
                40000,
                45000,
                50000,
                60000,
            ],
            "TARGET": [
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        }
    )

    result = calculate_woe_iv(
        df,
        "income",
        target="TARGET",
    )

    assert result is not None

    grouped, iv, bin_edges, binning_type = result

    assert binning_type == "continuous"

    assert isinstance(
        grouped,
        pd.DataFrame,
    )

    assert isinstance(
        iv,
        (float, np.floating),
    )

    assert iv >= 0

    assert bin_edges is not None

    assert "woe" in grouped.columns
    assert "iv" in grouped.columns
    assert "good" in grouped.columns
    assert "bad" in grouped.columns


# ============================================================
# CATEGORICAL FEATURE
# ============================================================

def test_categorical_feature_woe_iv():

    df = pd.DataFrame(
        {
            "education": [
                "Graduate",
                "Graduate",
                "Graduate",
                "High School",
                "High School",
                "High School",
                "Masters",
                "Masters",
                "Other",
                "Other",
            ],
            "TARGET": [
                0,
                0,
                1,
                1,
                1,
                0,
                0,
                0,
                1,
                0,
            ],
        }
    )

    result = calculate_woe_iv(
        df,
        "education",
        target="TARGET",
    )

    assert result is not None

    grouped, iv, bin_edges, binning_type = result

    assert binning_type == "low_cardinality"

    assert bin_edges is None

    assert iv >= 0

    assert "woe" in grouped.columns
    assert "iv" in grouped.columns


# ============================================================
# MISSING CATEGORICAL VALUES
# ============================================================

def test_categorical_missing_values_get_missing_bin():

    df = pd.DataFrame(
        {
            "occupation": [
                "Engineer",
                None,
                "Teacher",
                "Engineer",
                np.nan,
                "Doctor",
            ],
            "TARGET": [
                0,
                1,
                0,
                1,
                0,
                0,
            ],
        }
    )

    result = calculate_woe_iv(
        df,
        "occupation",
        target="TARGET",
    )

    assert result is not None

    grouped, iv, bin_edges, binning_type = result

    bins = (
        grouped["bin"]
        .astype(str)
        .tolist()
    )

    assert "__MISSING__" in bins


# ============================================================
# MISSING NUMERIC VALUES
# ============================================================

def test_numeric_missing_values_get_missing_bin():

    df = pd.DataFrame(
        {
            "credit_score": [
                600,
                650,
                np.nan,
                700,
                750,
                np.nan,
                800,
                820,
                580,
                620,
            ],
            "TARGET": [
                1,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
            ],
        }
    )

    result = calculate_woe_iv(
        df,
        "credit_score",
        target="TARGET",
    )

    assert result is not None

    grouped, iv, bin_edges, binning_type = result

    bins = (
        grouped["bin"]
        .astype(str)
        .tolist()
    )

    assert "__MISSING__" in bins


# ============================================================
# WOE MATHEMATICAL VALIDATION
# ============================================================

def test_woe_values_are_finite():

    df = pd.DataFrame(
        {
            "feature": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
            ],
            "TARGET": [
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
            ],
        }
    )

    result = calculate_woe_iv(
        df,
        "feature",
        target="TARGET",
    )

    assert result is not None

    grouped, iv, _, _ = result

    assert np.isfinite(
        grouped["woe"]
    ).all()

    assert np.isfinite(iv)


# ============================================================
# INVALID NUMERIC BINNING
# ============================================================
def test_numeric_constant_feature_returns_zero_iv():

    df = pd.DataFrame(
        {
            "constant_feature": [
                10,
                10,
                10,
                10,
                10,
                10,
            ],
            "TARGET": [
                0,
                1,
                0,
                1,
                0,
                1,
            ],
        }
    )

    result = calculate_woe_iv(
        df,
        "constant_feature",
        target="TARGET",
    )

    assert result is not None

    grouped, iv, bin_edges, binning_type = result

    assert binning_type == "continuous"
    assert iv == 0.0

    assert np.isfinite(
        grouped["woe"]
    ).all()

    assert (
        grouped["woe"] == 0.0
    ).all()


# ============================================================
# TARGET VALIDATION
# ============================================================

def test_woe_requires_target_column():

    df = pd.DataFrame(
        {
            "income": [
                10000,
                20000,
                30000,
            ]
        }
    )

    with pytest.raises(
        KeyError
    ):

        calculate_woe_iv(
            df,
            "income",
            target="TARGET",
        )