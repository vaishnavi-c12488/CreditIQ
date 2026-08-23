"""
CreditIQ ETL Tests
"""

from pathlib import Path

import pandas as pd
import pytest

from src.data import etl


# ============================================================
# SQL FILE
# ============================================================

def test_final_etl_sql_exists():

    assert etl.SQL_FILE.exists()
    assert etl.SQL_FILE.is_file()


def test_final_etl_sql_is_not_empty():

    sql = etl.SQL_FILE.read_text(
        encoding="utf-8"
    ).strip()

    assert len(sql) > 0


# ============================================================
# OUTPUT PATH
# ============================================================

def test_output_directory_is_configured():

    assert isinstance(
        etl.OUTPUT_DIR,
        Path,
    )

    assert (
        etl.OUTPUT_FILE.name
        == "applicant_features_raw.parquet"
    )


# ============================================================
# ETL OUTPUT VALIDATION
# ============================================================

def test_etl_output_exists():

    assert etl.OUTPUT_FILE.exists()


def test_etl_output_schema():

    df = pd.read_parquet(
        etl.OUTPUT_FILE
    )

    assert "SK_ID_CURR" in df.columns

    assert len(df) == 307_511

    assert (
        df["SK_ID_CURR"].nunique()
        == 307_511
    )


# ============================================================
# ETL OUTPUT QUALITY
# ============================================================

def test_etl_applicant_ids_are_not_null():

    df = pd.read_parquet(
        etl.OUTPUT_FILE
    )

    assert (
        df["SK_ID_CURR"]
        .isna()
        .sum()
        == 0
    )


def test_etl_applicant_ids_are_positive():

    df = pd.read_parquet(
        etl.OUTPUT_FILE
    )

    assert (
        df["SK_ID_CURR"] > 0
    ).all()


# ============================================================
# ETL FAILURE VALIDATION
# ============================================================

def test_etl_rejects_wrong_row_count(
    monkeypatch,
):
    """
    Verify that the ETL raises ValueError
    when the SQL query returns an unexpected
    number of rows.
    """

    fake_df = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3],
        }
    )

    monkeypatch.setattr(
        etl.pd,
        "read_sql_query",
        lambda sql, engine: fake_df,
    )

    with pytest.raises(
        ValueError,
        match="Unexpected row count",
    ):

        etl.run_etl()