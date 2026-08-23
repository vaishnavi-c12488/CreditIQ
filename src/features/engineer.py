"""
CreditIQ - Feature Engineering

This script creates useful applicant-level features from the
raw applicant data before building the credit-risk models.

What we do:
- Handle the DAYS_EMPLOYED anomaly in the Home Credit data.
- Convert employment duration from days into years.
- Create income, credit, and annuity-based ratios.
- Capture previous application refusals.
- Create CREDIT_UTILIZATION using available bureau-level aggregates.
- Do not create ACTIVE_CREDIT_RATIO because the required underlying
  fields are not available in the current dataset.
- Keep the same applicants and validate that SK_ID_CURR stays unique.

The engineered dataset is saved and used by the next stages
of the CreditIQ pipeline.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "applicant_features_raw.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "applicant_features_engineered.parquet"
)


# ============================================================
# Feature Engineering
# ============================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    created_features = []

    # ========================================================
    # 1. DAYS_EMPLOYED anomaly
    # ========================================================

    if "DAYS_EMPLOYED" in df.columns:

        # Home Credit uses 365243 as an anomaly value.
        df["DAYS_EMPLOYED_ANOM"] = (
            df["DAYS_EMPLOYED"] == 365243
        ).astype(int)

        # Replace anomaly with missing value.
        df["DAYS_EMPLOYED"] = (
            df["DAYS_EMPLOYED"]
            .replace(365243, np.nan)
        )

        # Convert employment duration from days to years.
        df["YEARS_EMPLOYED"] = (
            -df["DAYS_EMPLOYED"] / 365.25
        )

        created_features.extend([
            "DAYS_EMPLOYED_ANOM",
            "YEARS_EMPLOYED"
        ])

    # ========================================================
    # 2. Debt-to-Income Ratio
    # ========================================================

    required_columns = {
        "AMT_ANNUITY",
        "AMT_INCOME_TOTAL"
    }

    if required_columns.issubset(df.columns):

        df["DEBT_TO_INCOME"] = (
            df["AMT_ANNUITY"]
            / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
        )

        created_features.append(
            "DEBT_TO_INCOME"
        )

    # ========================================================
    # 3. Credit-to-Income Ratio
    # ========================================================

    required_columns = {
        "AMT_CREDIT",
        "AMT_INCOME_TOTAL"
    }

    if required_columns.issubset(df.columns):

        df["CREDIT_TO_INCOME"] = (
            df["AMT_CREDIT"]
            / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
        )

        created_features.append(
            "CREDIT_TO_INCOME"
        )

    # ========================================================
    # 4. Annuity-to-Credit Ratio
    # ========================================================

    required_columns = {
        "AMT_ANNUITY",
        "AMT_CREDIT"
    }

    if required_columns.issubset(df.columns):

        df["ANNUITY_TO_CREDIT"] = (
            df["AMT_ANNUITY"]
            / df["AMT_CREDIT"].replace(0, np.nan)
        )

        created_features.append(
            "ANNUITY_TO_CREDIT"
        )

    # ========================================================
    # 5. Previous Application Refusal Count
    # ========================================================

    refusal_columns = [
        col
        for col in df.columns
        if "PREV" in col.upper()
        and "REFUS" in col.upper()
    ]

    if refusal_columns:

        df["PREVIOUS_REFUSAL_COUNT"] = (
            df[refusal_columns]
            .fillna(0)
            .sum(axis=1)
        )

        created_features.append(
            "PREVIOUS_REFUSAL_COUNT"
        )

    # ========================================================
    # 6. Credit Utilization
    #
    # Formula:
    #
    #     Total Bureau Debt
    #     -----------------
    #     Total Bureau Credit
    #
    # These are applicant-level aggregates already available
    # in the input dataset.
    # ========================================================

    required_columns = {
        "bureau_total_debt",
        "bureau_total_credit"
    }

    if required_columns.issubset(df.columns):

        df["CREDIT_UTILIZATION"] = (
            df["bureau_total_debt"]
            / df["bureau_total_credit"].replace(0, np.nan)
        )

        created_features.append(
            "CREDIT_UTILIZATION"
        )

    else:

        missing_columns = (
            required_columns - set(df.columns)
        )

        raise ValueError(
            "Cannot create CREDIT_UTILIZATION. "
            f"Missing columns: {sorted(missing_columns)}"
        )

    # ========================================================
    # 7. ACTIVE_CREDIT_RATIO
    #
    # This feature is intentionally NOT created.
    #
    # The original implementation expected:
    #
    #     BUREAU_ACTIVE_CREDIT
    #     BUREAU_TOTAL_CREDIT
    #
    # Those underlying fields are not available in the
    # current applicant-level dataset.
    #
    # We therefore do not fabricate or approximate this
    # feature without a defensible source definition.
    # ========================================================

    # ========================================================
    # Feature validation
    # ========================================================

    expected_features = [
        "DAYS_EMPLOYED_ANOM",
        "YEARS_EMPLOYED",
        "DEBT_TO_INCOME",
        "CREDIT_TO_INCOME",
        "ANNUITY_TO_CREDIT",
        "PREVIOUS_REFUSAL_COUNT",
        "CREDIT_UTILIZATION"
    ]

    missing_features = [
        feature
        for feature in expected_features
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Expected engineered features were not created: "
            f"{missing_features}"
        )

    print("\nCreated engineered features:")

    for feature in created_features:
        print(f"  [OK] {feature}")

    return df


# ============================================================
# Main Feature Engineering Pipeline
# ============================================================

def run_feature_engineering():

    print("=" * 60)
    print("CreditIQ FEATURE ENGINEERING")
    print("=" * 60)

    # ========================================================
    # Check input file
    # ========================================================

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # Load raw applicant features
    # ========================================================

    print("\nReading raw applicant features...")

    df = pd.read_parquet(
        INPUT_FILE
    )

    print(
        f"Input dataset: "
        f"{len(df):,} rows × "
        f"{len(df.columns):,} columns"
    )

    # ========================================================
    # Validate input ID
    # ========================================================

    if "SK_ID_CURR" not in df.columns:

        raise ValueError(
            "SK_ID_CURR is missing from input dataset."
        )

    if df["SK_ID_CURR"].nunique() != len(df):

        raise ValueError(
            "SK_ID_CURR is not unique in input dataset."
        )

    # ========================================================
    # Feature engineering
    # ========================================================

    print("\nCreating engineered features...")

    df_engineered = engineer_features(df)

    # ========================================================
    # Validate row count
    # ========================================================

    if len(df_engineered) != len(df):

        raise ValueError(
            "Row count changed during feature engineering."
        )

    print(
        "\nRow-count validation passed."
    )

    # ========================================================
    # Validate ID uniqueness
    # ========================================================

    if (
        df_engineered["SK_ID_CURR"].nunique()
        != len(df_engineered)
    ):

        raise ValueError(
            "SK_ID_CURR is not unique after feature engineering."
        )

    print(
        "SK_ID_CURR uniqueness validation passed."
    )

    # ========================================================
    # Important feature verification
    # ========================================================

    print("\nImportant feature verification:")

    print(
        "  CREDIT_UTILIZATION:",
        "CREDIT_UTILIZATION" in df_engineered.columns
    )

    print(
        "  ACTIVE_CREDIT_RATIO:",
        "ACTIVE_CREDIT_RATIO" in df_engineered.columns
    )

    # ========================================================
    # Save engineered dataset
    # ========================================================

    df_engineered.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nEngineered dataset: "
        f"{len(df_engineered):,} rows × "
        f"{len(df_engineered.columns):,} columns"
    )

    print(
        f"\nSaved engineered features to:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        "\nCreditIQ feature engineering completed successfully."
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    run_feature_engineering()