"""
CreditIQ - WOE/IV Feature Engineering

This script prepares features for the CreditIQ credit-risk models.

Pipeline:
1. Load train, validation, and test datasets.
2. Calculate WOE and IV using TRAIN ONLY.
3. Treat binary/low-cardinality numeric variables as discrete bins.
4. Treat continuous numeric variables using training-derived quantile bins.
5. Apply the TRAIN WOE mapping to validation and test.
6. Handle missing and out-of-range/unseen values consistently.
7. Select features using IV >= 0.02.
8. Save complete WOE datasets and final model-ready datasets.

Important:
- Validation and test data are NEVER used to calculate WOE/IV.
- Bin definitions are learned from TRAIN only.
- WOE mappings are learned from TRAIN only.
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

    bin_edges = None
    binning_type = None

    # ========================================================
    # Numeric features
    # ========================================================

    if pd.api.types.is_numeric_dtype(data[feature]):

        unique_values = data[feature].dropna().unique()

        # ----------------------------------------------------
        # Binary numeric feature
        # ----------------------------------------------------

        if len(unique_values) <= 2:

            binning_type = "binary"

            temp["bin"] = temp[feature].astype(object)

            temp.loc[
                data[feature].isna(),
                "bin"
            ] = "__MISSING__"

        # ----------------------------------------------------
        # Low-cardinality numeric feature
        # ----------------------------------------------------

        elif len(unique_values) <= 10:

            binning_type = "low_cardinality"

            temp["bin"] = temp[feature].astype(object)

            temp.loc[
                data[feature].isna(),
                "bin"
            ] = "__MISSING__"

        # ----------------------------------------------------
        # Continuous numeric feature
        # ----------------------------------------------------

        else:

            binning_type = "continuous"

            try:

                temp["bin"], bin_edges = pd.qcut(
                    data[feature],
                    q=10,
                    duplicates="drop",
                    retbins=True
                )

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

        binning_type = "categorical"

        temp["bin"] = (
            temp[feature]
            .astype(str)
            .replace("nan", "__MISSING__")
        )

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

    return grouped, iv, bin_edges, binning_type


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
        # WOE / IV calculated ONLY using TRAIN
        # ----------------------------------------------------

        result = calculate_woe_iv(
            train,
            feature,
            target="TARGET"
        )

        if result is None:

            print("  Skipped.")
            continue

        grouped, iv, bin_edges, binning_type = result

        iv_results.append(
            {
                "feature": feature,
                "iv": iv,
                "binning_type": binning_type
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

                # ------------------------------------------------
                # Binary / low-cardinality numeric feature
                # ------------------------------------------------

                if binning_type in {
                    "binary",
                    "low_cardinality"
                }:

                    train_keys = (
                        train[feature]
                        .astype(object)
                        .where(
                            train[feature].notna(),
                            "__MISSING__"
                        )
                        .astype(str)
                    )

                    validation_keys = (
                        validation[feature]
                        .astype(object)
                        .where(
                            validation[feature].notna(),
                            "__MISSING__"
                        )
                        .astype(str)
                    )

                    test_keys = (
                        test[feature]
                        .astype(object)
                        .where(
                            test[feature].notna(),
                            "__MISSING__"
                        )
                        .astype(str)
                    )

                # ------------------------------------------------
                # Continuous numeric feature
                # ------------------------------------------------

                else:

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

                    # --------------------------------------------
                    # Detect values outside training range
                    # --------------------------------------------

                    train_out_of_range = (
                        train[feature].notna()
                        &
                        train_bins.isna()
                    )

                    validation_out_of_range = (
                        validation[feature].notna()
                        &
                        validation_bins.isna()
                    )

                    test_out_of_range = (
                        test[feature].notna()
                        &
                        test_bins.isna()
                    )

                    train_keys.loc[
                        train_out_of_range
                    ] = "__OUT_OF_RANGE__"

                    validation_keys.loc[
                        validation_out_of_range
                    ] = "__OUT_OF_RANGE__"

                    test_keys.loc[
                        test_out_of_range
                    ] = "__OUT_OF_RANGE__"

                # ------------------------------------------------
                # Apply TRAIN WOE mapping
                # ------------------------------------------------

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
                # Missing-value WOE
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

                # ------------------------------------------------
                # Out-of-range fallback
                #
                # Learned only from TRAIN.
                # ------------------------------------------------

                if binning_type == "continuous":

                    out_of_range_woe = (
                        grouped.loc[
                            grouped["bin"].astype(str)
                            != "__MISSING__",
                            "woe"
                        ].mean()
                    )

                    if pd.notna(out_of_range_woe):

                        train_woe.loc[
                            train_keys == "__OUT_OF_RANGE__",
                            feature
                        ] = out_of_range_woe

                        validation_woe.loc[
                            validation_keys == "__OUT_OF_RANGE__",
                            feature
                        ] = out_of_range_woe

                        test_woe.loc[
                            test_keys == "__OUT_OF_RANGE__",
                            feature
                        ] = out_of_range_woe

            except (ValueError, TypeError):

                print("  Mapping failed. Skipped.")
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
    # Final unmapped-value handling
    #
    # Any remaining NaN means:
    # - unseen validation/test category
    # - unexpected value
    # - unavailable mapping
    #
    # We use neutral WOE = 0.0.
    #
    # This does NOT learn anything from validation/test.
    # ========================================================

    print(
        "\nApplying neutral WOE fallback "
        "for remaining unmapped values..."
    )

    for feature in features:

        if feature in train_woe.columns:

            train_woe[feature] = (
                train_woe[feature]
                .fillna(0.0)
            )

            validation_woe[feature] = (
                validation_woe[feature]
                .fillna(0.0)
            )

            test_woe[feature] = (
                test_woe[feature]
                .fillna(0.0)
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
    # WOE NaN check
    # ========================================================

    woe_features = [
        feature
        for feature in features
        if feature in train_woe.columns
    ]

    train_nan_count = (
        train_woe[woe_features]
        .isna()
        .sum()
        .sum()
    )

    validation_nan_count = (
        validation_woe[woe_features]
        .isna()
        .sum()
        .sum()
    )

    test_nan_count = (
        test_woe[woe_features]
        .isna()
        .sum()
        .sum()
    )

    print("\nWOE NaN check:")
    print(
        f"Train      : {train_nan_count:,}"
    )
    print(
        f"Validation : {validation_nan_count:,}"
    )
    print(
        f"Test       : {test_nan_count:,}"
    )

    if (
        train_nan_count == 0
        and validation_nan_count == 0
        and test_nan_count == 0
    ):

        print(
            "PASS: No NaN values in WOE features."
        )

    else:

        print(
            "WARNING: Some WOE values remain NaN."
        )

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

    # ========================================================
    # Save selected features
    # ========================================================

    selected_report = (
        iv_report[
            iv_report["feature"].isin(
                selected_features
            )
        ][
            ["feature", "iv"]
        ]
        .copy()
    )

    selected_report.to_csv(
        SELECTED_FEATURES_REPORT,
        index=False
    )

    # ========================================================
    # Save removed features
    # ========================================================

    removed_report = (
        iv_report[
            iv_report["feature"].isin(
                removed_features
            )
        ][
            ["feature", "iv"]
        ]
        .copy()
    )

    removed_report.to_csv(
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

    expected_features = len(
        selected_features
    )

    expected_columns = (
        expected_features + 2
    )

    if train_final.shape[1] != expected_columns:

        raise ValueError(
            "Unexpected number of columns in train_final."
        )

    if validation_final.shape[1] != expected_columns:

        raise ValueError(
            "Unexpected number of columns in validation_final."
        )

    if test_final.shape[1] != expected_columns:

        raise ValueError(
            "Unexpected number of columns in test_final."
        )

    if train_final["SK_ID_CURR"].nunique() != len(
        train_final
    ):

        raise ValueError(
            "SK_ID_CURR is not unique in train_final."
        )

    if validation_final["SK_ID_CURR"].nunique() != len(
        validation_final
    ):

        raise ValueError(
            "SK_ID_CURR is not unique in validation_final."
        )

    if test_final["SK_ID_CURR"].nunique() != len(
        test_final
    ):

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

    print("\nBinning summary:")

    if "binning_type" in iv_report.columns:

        print(
            iv_report[
                "binning_type"
            ]
            .value_counts()
            .to_string()
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