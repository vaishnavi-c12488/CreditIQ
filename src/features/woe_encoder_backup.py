"""
CreditIQ - WOE/IV Feature Engineering

This script prepares the features for our credit-risk models.

What we do:
- Load the time-based train, validation, and test datasets.
- Calculate WOE and IV using only the training data.
- Apply the same WOE mapping to validation and test data.
- Use IV to identify the most useful features.
- Keep features with IV >= 0.02 and remove weaker ones.
- Save the WOE datasets and the final model-ready datasets.

We started with 183 features:
79 were selected and 104 were removed.

The final datasets contain 79 WOE features,
along with SK_ID_CURR and TARGET.

WOE/IV is calculated only on training data
to avoid data leakage.
"""


from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "train.parquet"
VAL_FILE = PROJECT_ROOT / "data" / "processed" / "validation.parquet"
TEST_FILE = PROJECT_ROOT / "data" / "processed" / "test.parquet"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_OUTPUT = OUTPUT_DIR / "train_woe.parquet"
VAL_OUTPUT = OUTPUT_DIR / "validation_woe.parquet"
TEST_OUTPUT = OUTPUT_DIR / "test_woe.parquet"

TRAIN_FINAL_OUTPUT = OUTPUT_DIR / "train_final.parquet"
VAL_FINAL_OUTPUT = OUTPUT_DIR / "validation_final.parquet"
TEST_FINAL_OUTPUT = OUTPUT_DIR / "test_final.parquet"

IV_REPORT = OUTPUT_DIR / "woe_iv_report.csv"
SELECTED_FEATURES_REPORT = OUTPUT_DIR / "selected_features.csv"
REMOVED_FEATURES_REPORT = OUTPUT_DIR / "removed_features.csv"


# ============================================================
# WOE / IV calculation
# ============================================================

def calculate_woe_iv(data, feature, target="TARGET"):

    temp = data[[feature, target]].copy()

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    temp[feature] = temp[feature].where(
        temp[feature].notna(),
        "__MISSING__"
    )

    # ========================================================
    # Numeric features
    # ========================================================

    if pd.api.types.is_numeric_dtype(data[feature]):

        try:
            temp["bin"], bin_edges = pd.qcut(
                data[feature],
                q=10,
                duplicates="drop",
                retbins=True
            )

            # Missing values get their own group
            temp["bin"] = temp["bin"].astype(object)

            temp.loc[
                data[feature].isna(),
                "bin"
            ] = "__MISSING__"

        except (ValueError, TypeError):
            return None

    # ========================================================
    # Categorical features
    # ========================================================

    else:

        temp["bin"] = (
            temp[feature]
            .astype(str)
            .replace("nan", "__MISSING__")
        )

        bin_edges = None

    # ========================================================
    # Good / Bad counts
    #
    # TARGET = 0 → Good
    # TARGET = 1 → Bad
    # ========================================================

    grouped = (
        temp.groupby(
            "bin",
            dropna=False,
            observed=False
        )[target]
        .agg(
            total="count",
            bad="sum"
        )
        .reset_index()
    )

    grouped["good"] = (
        grouped["total"] - grouped["bad"]
    )

    # ========================================================
    # Good / Bad distributions
    # ========================================================

    epsilon = 1e-6

    total_good = grouped["good"].sum()
    total_bad = grouped["bad"].sum()

    grouped["dist_good"] = (
        (grouped["good"] + epsilon)
        /
        (total_good + epsilon * len(grouped))
    )

    grouped["dist_bad"] = (
        (grouped["bad"] + epsilon)
        /
        (total_bad + epsilon * len(grouped))
    )

    # ========================================================
    # WOE
    # ========================================================

    grouped["woe"] = np.log(
        grouped["dist_good"]
        /
        grouped["dist_bad"]
    )

    # ========================================================
    # IV
    # ========================================================

    grouped["iv"] = (
        grouped["dist_good"]
        -
        grouped["dist_bad"]
    ) * grouped["woe"]

    iv = grouped["iv"].sum()

    return grouped, iv, bin_edges


# ============================================================
# Main WOE pipeline
# ============================================================

def run_woe():

    print("Starting CreditIQ WOE/IV encoding...")

    # ========================================================
    # Load datasets
    # ========================================================

    train = pd.read_parquet(TRAIN_FILE)
    validation = pd.read_parquet(VAL_FILE)
    test = pd.read_parquet(TEST_FILE)

    print(
        f"Train: {len(train):,} rows × "
        f"{len(train.columns):,} columns"
    )

    print(
        f"Validation: {len(validation):,} rows × "
        f"{len(validation.columns):,} columns"
    )

    print(
        f"Test: {len(test):,} rows × "
        f"{len(test.columns):,} columns"
    )

    # ========================================================
    # Exclude ID and TARGET
    # ========================================================

    exclude_columns = {
        "SK_ID_CURR",
        "TARGET"
    }

    features = [
        col
        for col in train.columns
        if col not in exclude_columns
    ]

    print(
        f"Features to process: {len(features)}"
    )

    # ========================================================
    # Create WOE copies
    # ========================================================

    train_woe = train.copy()
    validation_woe = validation.copy()
    test_woe = test.copy()

    iv_results = []

    # ========================================================
    # Process every feature
    # ========================================================

    for i, feature in enumerate(
        features,
        start=1
    ):

        print(
            f"[{i}/{len(features)}] {feature}"
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # WOE / IV is calculated ONLY using TRAIN
        # ----------------------------------------------------

        result = calculate_woe_iv(
            train,
            feature,
            target="TARGET"
        )

        if result is None:
            print("  Skipped.")
            continue

        grouped, iv, bin_edges = result

        iv_results.append(
            {
                "feature": feature,
                "iv": iv
            }
        )

        # ----------------------------------------------------
        # WOE mapping learned from TRAIN
        # ----------------------------------------------------

        mapping = dict(
            zip(
                grouped["bin"].astype(str),
                grouped["woe"]
            )
        )

        # ====================================================
        # Numeric feature
        # ====================================================

        if pd.api.types.is_numeric_dtype(
            train[feature]
        ):

            try:

                train_bins = pd.cut(
                    train[feature],
                    bins=bin_edges,
                    include_lowest=True
                )

                validation_bins = pd.cut(
                    validation[feature],
                    bins=bin_edges,
                    include_lowest=True
                )

                test_bins = pd.cut(
                    test[feature],
                    bins=bin_edges,
                    include_lowest=True
                )

                train_keys = train_bins.astype(str)
                validation_keys = validation_bins.astype(str)
                test_keys = test_bins.astype(str)

                train_woe[feature] = (
                    train_keys.map(mapping)
                )

                validation_woe[feature] = (
                    validation_keys.map(mapping)
                )

                test_woe[feature] = (
                    test_keys.map(mapping)
                )

                # ------------------------------------------------
                # Apply missing-value WOE
                # ------------------------------------------------

                missing_woe = mapping.get(
                    "__MISSING__"
                )

                if missing_woe is not None:

                    train_woe.loc[
                        train[feature].isna(),
                        feature
                    ] = missing_woe

                    validation_woe.loc[
                        validation[feature].isna(),
                        feature
                    ] = missing_woe

                    test_woe.loc[
                        test[feature].isna(),
                        feature
                    ] = missing_woe

            except (ValueError, TypeError):
                continue

        # ====================================================
        # Categorical feature
        # ====================================================

        else:

            train_keys = (
                train[feature]
                .astype(str)
                .replace("nan", "__MISSING__")
            )

            validation_keys = (
                validation[feature]
                .astype(str)
                .replace("nan", "__MISSING__")
            )

            test_keys = (
                test[feature]
                .astype(str)
                .replace("nan", "__MISSING__")
            )

            train_woe[feature] = (
                train_keys.map(mapping)
            )

            validation_woe[feature] = (
                validation_keys.map(mapping)
            )

            test_woe[feature] = (
                test_keys.map(mapping)
            )

    # ========================================================
    # Create IV report
    # ========================================================

    iv_report = pd.DataFrame(
        iv_results
    )

    iv_report = iv_report.sort_values(
        "iv",
        ascending=False
    )

    iv_report.to_csv(
        IV_REPORT,
        index=False
    )

    print("\nIV report created.")

    # ========================================================
    # Save complete WOE datasets
    # ========================================================

    train_woe.to_parquet(
        TRAIN_OUTPUT,
        index=False
    )

    validation_woe.to_parquet(
        VAL_OUTPUT,
        index=False
    )

    test_woe.to_parquet(
        TEST_OUTPUT,
        index=False
    )

    print("Complete WOE datasets saved.")

    # ========================================================
    # IV-based feature selection
    # ========================================================

    IV_THRESHOLD = 0.02

    selected_features = (
        iv_report.loc[
            iv_report["iv"] >= IV_THRESHOLD,
            "feature"
        ]
        .tolist()
    )

    removed_features = (
        iv_report.loc[
            iv_report["iv"] < IV_THRESHOLD,
            "feature"
        ]
        .tolist()
    )

    # --------------------------------------------------------
    # Save selected and removed feature lists
    # --------------------------------------------------------

    pd.DataFrame(
        {
            "feature": selected_features,
            "iv": iv_report.set_index(
                "feature"
            ).loc[
                selected_features,
                "iv"
            ].values
        }
    ).to_csv(
        SELECTED_FEATURES_REPORT,
        index=False
    )

    pd.DataFrame(
        {
            "feature": removed_features,
            "iv": iv_report.set_index(
                "feature"
            ).loc[
                removed_features,
                "iv"
            ].values
        }
    ).to_csv(
        REMOVED_FEATURES_REPORT,
        index=False
    )

    # ========================================================
    # Create final model-ready feature list
    # ========================================================

    final_columns = (
        [
            "SK_ID_CURR",
            "TARGET"
        ]
        +
        selected_features
    )

    train_final = train_woe[
        final_columns
    ].copy()

    validation_final = validation_woe[
        final_columns
    ].copy()

    test_final = test_woe[
        final_columns
    ].copy()

    # ========================================================
    # Save final model-ready datasets
    # ========================================================

    train_final.to_parquet(
        TRAIN_FINAL_OUTPUT,
        index=False
    )

    validation_final.to_parquet(
        VAL_FINAL_OUTPUT,
        index=False
    )

    test_final.to_parquet(
        TEST_FINAL_OUTPUT,
        index=False
    )

    # ========================================================
    # Final validation
    # ========================================================

    expected_features = len(selected_features)

    expected_columns = (
        expected_features + 2
    )

    if train_final.shape[1] != expected_columns:
        raise ValueError(
            "Unexpected number of columns in "
            "train_final."
        )

    if validation_final.shape[1] != expected_columns:
        raise ValueError(
            "Unexpected number of columns in "
            "validation_final."
        )

    if test_final.shape[1] != expected_columns:
        raise ValueError(
            "Unexpected number of columns in "
            "test_final."
        )

    if train_final["SK_ID_CURR"].nunique() != len(train_final):
        raise ValueError(
            "SK_ID_CURR is not unique in train_final."
        )

    if validation_final["SK_ID_CURR"].nunique() != len(
        validation_final
    ):
        raise ValueError(
            "SK_ID_CURR is not unique in validation_final."
        )

    if test_final["SK_ID_CURR"].nunique() != len(test_final):
        raise ValueError(
            "SK_ID_CURR is not unique in test_final."
        )

    # ========================================================
    # Summary
    # ========================================================

    print("\n" + "=" * 60)
    print("TOP 20 FEATURES BY IV")
    print("=" * 60)

    print(
        iv_report.head(20).to_string(
            index=False
        )
    )

    print("\n" + "=" * 60)
    print("IV FEATURE SELECTION")
    print("=" * 60)

    print(
        f"Total features evaluated: "
        f"{len(iv_report)}"
    )

    print(
        f"IV threshold: "
        f"{IV_THRESHOLD}"
    )

    print(
        f"Features selected: "
        f"{len(selected_features)}"
    )

    print(
        f"Features removed: "
        f"{len(removed_features)}"
    )

    print("\nFinal model-ready datasets:")

    print(
        f"TRAIN: "
        f"{train_final.shape}"
    )

    print(
        f"VALIDATION: "
        f"{validation_final.shape}"
    )

    print(
        f"TEST: "
        f"{test_final.shape}"
    )

    print("\nSaved files:")

    print(TRAIN_OUTPUT)
    print(VAL_OUTPUT)
    print(TEST_OUTPUT)
    print(IV_REPORT)
    print(SELECTED_FEATURES_REPORT)
    print(REMOVED_FEATURES_REPORT)
    print(TRAIN_FINAL_OUTPUT)
    print(VAL_FINAL_OUTPUT)
    print(TEST_FINAL_OUTPUT)

    print(
        "\nCreditIQ WOE/IV pipeline "
        "completed successfully."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    run_woe()