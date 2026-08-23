"""
CreditIQ — Feature Store

Provides a controlled interface for loading and validating
applicant-level engineered features.

The feature store does not perform feature engineering.
Feature engineering is handled by engineer.py.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "applicant_features_engineered.parquet"
)


# ============================================================
# Required identifiers
# ============================================================

ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"


# ============================================================
# Feature Store
# ============================================================

class CreditIQFeatureStore:
    """
    Access layer for CreditIQ applicant-level engineered features.
    """

    def __init__(self, feature_file: Path = FEATURE_FILE):
        self.feature_file = Path(feature_file)

    # --------------------------------------------------------
    # Load feature data
    # --------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """
        Load the engineered applicant feature dataset.
        """

        if not self.feature_file.exists():
            raise FileNotFoundError(
                f"Feature store file not found:\n"
                f"{self.feature_file}"
            )

        df = pd.read_parquet(self.feature_file)

        self.validate(df)

        return df

    # --------------------------------------------------------
    # Validate feature store
    # --------------------------------------------------------

    @staticmethod
    def validate(df: pd.DataFrame) -> None:
        """
        Validate the structural integrity of the feature store.
        """

        if ID_COLUMN not in df.columns:
            raise ValueError(
                f"Required identifier '{ID_COLUMN}' is missing."
            )

        if TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"Required target '{TARGET_COLUMN}' is missing."
            )

        if df[ID_COLUMN].isna().any():
            raise ValueError(
                f"{ID_COLUMN} contains missing values."
            )

        if df[ID_COLUMN].duplicated().any():
            raise ValueError(
                f"{ID_COLUMN} contains duplicate applicants."
            )

        if len(df) == 0:
            raise ValueError(
                "Feature store is empty."
            )

    # --------------------------------------------------------
    # Get model features
    # --------------------------------------------------------

    def get_model_features(
        self,
        selected_features: list[str],
    ) -> pd.DataFrame:
        """
        Return only the requested model features.

        SK_ID_CURR and TARGET are excluded from the model matrix.
        """

        df = self.load()

        missing_features = [
            feature
            for feature in selected_features
            if feature not in df.columns
        ]

        if missing_features:
            raise ValueError(
                "Requested model features are missing from "
                f"the feature store: {missing_features}"
            )

        return df[selected_features].copy()

    # --------------------------------------------------------
    # Get applicant
    # --------------------------------------------------------

    def get_applicant(
        self,
        applicant_id: int,
    ) -> pd.DataFrame:
        """
        Retrieve a single applicant by SK_ID_CURR.
        """

        df = self.load()

        applicant = df[
            df[ID_COLUMN] == applicant_id
        ].copy()

        if applicant.empty:
            raise KeyError(
                f"Applicant {applicant_id} was not found."
            )

        return applicant

    # --------------------------------------------------------
    # Feature metadata
    # --------------------------------------------------------

    def metadata(self) -> dict:
        """
        Return basic feature-store metadata.
        """

        df = self.load()

        return {
            "path": str(self.feature_file),
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "unique_applicants": int(
                df[ID_COLUMN].nunique()
            ),
            "target_present": TARGET_COLUMN in df.columns,
            "feature_columns": [
                column
                for column in df.columns
                if column not in [
                    ID_COLUMN,
                    TARGET_COLUMN,
                ]
            ],
        }


# ============================================================
# Convenience function
# ============================================================

def load_feature_store() -> pd.DataFrame:
    """
    Convenience function for loading the CreditIQ feature store.
    """

    store = CreditIQFeatureStore()

    return store.load()


# ============================================================
# Command-line validation
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CreditIQ FEATURE STORE")
    print("=" * 60)

    store = CreditIQFeatureStore()

    df = store.load()

    metadata = store.metadata()

    print("\nFeature store validation passed.")

    print(
        f"Applicants: {metadata['unique_applicants']:,}"
    )

    print(
        f"Columns: {metadata['columns']:,}"
    )

    print(
        f"Model-available features: "
        f"{len(metadata['feature_columns']):,}"
    )

    print(
        f"\nFeature store:\n"
        f"{metadata['path']}"
    )

    print("\nCreditIQ feature store validation completed.")