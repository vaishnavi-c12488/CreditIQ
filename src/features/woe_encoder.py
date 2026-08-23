"""
CreditIQ - WOE/IV Feature Engineering

Pipeline:
1. Load frozen train, validation, and test datasets.
2. Calculate WOE/IV using TRAIN data only.
3. Apply TRAIN WOE mappings to validation and test.
4. Correctly handle None/NaN categorical values as __MISSING__.
5. Audit unmapped values before neutral fallback.
6. Apply neutral WOE = 0.0 only where genuinely unmapped.
7. Select features using IV >= 0.02.
8. Save WOE and final model-ready datasets.

Important:
- Train/validation/test split is already frozen.
- Validation and test are never used to calculate WOE/IV.
- Missing categorical values receive their learned TRAIN WOE.
- Neutral fallback is only for genuinely unseen/unmapped values.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train.parquet"
)

VAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation.parquet"
)

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

TRAIN_OUTPUT = OUTPUT_DIR / "train_woe.parquet"
VAL_OUTPUT = OUTPUT_DIR / "validation_woe.parquet"
TEST_OUTPUT = OUTPUT_DIR / "test_woe.parquet"

TRAIN_FINAL_OUTPUT = OUTPUT_DIR / "train_final.parquet"
VAL_FINAL_OUTPUT = OUTPUT_DIR / "validation_final.parquet"
TEST_FINAL_OUTPUT = OUTPUT_DIR / "test_final.parquet"

IV_REPORT = OUTPUT_DIR / "woe_iv_report.csv"
SELECTED_FEATURES_REPORT = OUTPUT_DIR / "selected_features.csv"
REMOVED_FEATURES_REPORT = OUTPUT_DIR / "removed_features.csv"
FALLBACK_REPORT = OUTPUT_DIR / "woe_fallback_report.csv"


# ============================================================
# WOE / IV CALCULATION
# ============================================================

def calculate_woe_iv(
    data,
    feature,
    target="TARGET"
):

    temp = data[[feature, target]].copy()

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

            temp["bin"] = (
                temp["bin"]
                .astype(object)
            )

            # IMPORTANT:
            # Missing numeric values get their own WOE bin.
            temp.loc[
                data[feature].isna(),
                "bin"
            ] = "__MISSING__"

        except (ValueError, TypeError):

            return None

        binning_type = "continuous"

    # ========================================================
    # Categorical features
    # ========================================================

    else:

        # IMPORTANT FIX:
        # Handle None/NaN BEFORE converting to string.
        #
        # None -> __MISSING__
        # NaN  -> __MISSING__
        #
        # This makes the TRAIN mapping consistent with
        # validation and test.

        temp["bin"] = (
            temp[feature]
            .fillna("__MISSING__")
            .astype(str)
        )

        bin_edges = None

        unique_count = (
            temp["bin"]
            .nunique()
        )

        if unique_count <= 2:

            binning_type = "binary"

        elif unique_count <= 10:

            binning_type = "low_cardinality"

        else:

            binning_type = "categorical"

    # ========================================================
    # GOOD / BAD COUNTS
    #
    # TARGET = 0 -> Good
    # TARGET = 1 -> Bad
    # ========================================================

    grouped = (
        temp
        .groupby(
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
        grouped["total"]
        - grouped["bad"]
    )

    # ========================================================
    # GOOD / BAD DISTRIBUTIONS
    # ========================================================

    epsilon = 1e-6

    total_good = (
        grouped["good"].sum()
    )

    total_bad = (
        grouped["bad"].sum()
    )

    grouped["dist_good"] = (
        (grouped["good"] + epsilon)
        /
        (
            total_good
            + epsilon * len(grouped)
        )
    )

    grouped["dist_bad"] = (
        (grouped["bad"] + epsilon)
        /
        (
            total_bad
            + epsilon * len(grouped)
        )
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

    iv = (
        grouped["iv"]
        .sum()
    )

    return (
        grouped,
        iv,
        bin_edges,
        binning_type
    )


# ============================================================
# MAIN WOE PIPELINE
# ============================================================

def run_woe():

    print(
        "Starting CreditIQ WOE/IV encoding..."
    )

    # ========================================================
    # LOAD DATASETS
    # ========================================================

    train = pd.read_parquet(
        TRAIN_FILE
    )

    validation = pd.read_parquet(
        VAL_FILE
    )

    test = pd.read_parquet(
        TEST_FILE
    )

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
    # EXCLUDE ID AND TARGET
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
        f"Features to process: "
        f"{len(features)}"
    )

    # ========================================================
    # CREATE WOE COPIES
    # ========================================================

    train_woe = train.copy()
    validation_woe = validation.copy()
    test_woe = test.copy()

    iv_results = []

    # ========================================================
    # PROCESS FEATURES
    # ========================================================

    for i, feature in enumerate(
        features,
        start=1
    ):

        print(
            f"[{i}/{len(features)}] "
            f"{feature}"
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

        (
            grouped,
            iv,
            bin_edges,
            binning_type
        ) = result

        iv_results.append(
            {
                "feature": feature,
                "iv": iv,
                "binning_type": binning_type
            }
        )

        # ----------------------------------------------------
        # CREATE TRAIN WOE MAPPING
        # ----------------------------------------------------

        mapping = dict(
            zip(
                grouped["bin"]
                .astype(str),
                grouped["woe"]
            )
        )

        # ====================================================
        # NUMERIC FEATURE
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

                train_keys = (
                    train_bins
                    .astype(str)
                )

                validation_keys = (
                    validation_bins
                    .astype(str)
                )

                test_keys = (
                    test_bins
                    .astype(str)
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

                # ------------------------------------------------
                # MISSING NUMERIC VALUE
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
                # OUT-OF-RANGE FALLBACK
                #
                # Learned only from TRAIN.
                # ------------------------------------------------

                numeric_woe = (
                    grouped.loc[
                        grouped["bin"]
                        .astype(str)
                        != "__MISSING__",
                        "woe"
                    ]
                )

                if not numeric_woe.empty:

                    out_of_range_woe = (
                        numeric_woe.mean()
                    )

                    train_woe[feature] = (
                        train_woe[feature]
                        .fillna(
                            out_of_range_woe
                        )
                    )

                    validation_woe[feature] = (
                        validation_woe[feature]
                        .fillna(
                            out_of_range_woe
                        )
                    )

                    test_woe[feature] = (
                        test_woe[feature]
                        .fillna(
                            out_of_range_woe
                        )
                    )

            except (
                ValueError,
                TypeError
            ):

                continue

        # ====================================================
        # CATEGORICAL FEATURE
        # ====================================================

        else:

            # =================================================
            # IMPORTANT FIX
            #
            # Fill missing values BEFORE astype(str).
            #
            # Previous:
            # None -> "None"
            #
            # Correct:
            # None -> "__MISSING__"
            # =================================================

            train_keys = (
                train[feature]
                .fillna("__MISSING__")
                .astype(str)
            )

            validation_keys = (
                validation[feature]
                .fillna("__MISSING__")
                .astype(str)
            )

            test_keys = (
                test[feature]
                .fillna("__MISSING__")
                .astype(str)
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
    # FALLBACK AUDIT
    #
    # IMPORTANT:
    # We measure NaNs BEFORE replacing them with 0.0.
    # ========================================================

    print(
        "\nAuditing unmapped WOE values "
        "before neutral fallback..."
    )

    fallback_counts = []

    for feature in features:

        if feature not in train_woe.columns:

            continue

        train_count = int(
            train_woe[feature]
            .isna()
            .sum()
        )

        validation_count = int(
            validation_woe[feature]
            .isna()
            .sum()
        )

        test_count = int(
            test_woe[feature]
            .isna()
            .sum()
        )

        total_count = (
            train_count
            + validation_count
            + test_count
        )

        if total_count > 0:

            fallback_counts.append(
                {
                    "feature": feature,
                    "train_fallback": train_count,
                    "validation_fallback": validation_count,
                    "test_fallback": test_count,
                    "total_fallback": total_count
                }
            )

    fallback_report = pd.DataFrame(
        fallback_counts,
        columns=[
            "feature",
            "train_fallback",
            "validation_fallback",
            "test_fallback",
            "total_fallback"
        ]
    )

    # ========================================================
    # PRINT FALLBACK SUMMARY
    # ========================================================

    print(
        "\n============================================================"
    )

    print(
        "WOE FALLBACK AUDIT"
    )

    print(
        "============================================================"
    )

    if not fallback_report.empty:

        fallback_report = (
            fallback_report
            .sort_values(
                "total_fallback",
                ascending=False
            )
        )

        total_train_fallback = int(
            fallback_report[
                "train_fallback"
            ].sum()
        )

        total_validation_fallback = int(
            fallback_report[
                "validation_fallback"
            ].sum()
        )

        total_test_fallback = int(
            fallback_report[
                "test_fallback"
            ].sum()
        )

        print(
            f"Train fallback values      : "
            f"{total_train_fallback:,}"
        )

        print(
            f"Validation fallback values : "
            f"{total_validation_fallback:,}"
        )

        print(
            f"Test fallback values       : "
            f"{total_test_fallback:,}"
        )

        print(
            "\nTop features requiring fallback:"
        )

        print(
            fallback_report
            .head(20)
            .to_string(
                index=False
            )
        )

    else:

        print(
            "PASS: No unmapped WOE values "
            "were found before fallback."
        )

    # ========================================================
    # SAVE FALLBACK AUDIT
    # ========================================================

    fallback_report.to_csv(
        FALLBACK_REPORT,
        index=False
    )

    print(
        f"\nSaved fallback audit to:\n"
        f"{FALLBACK_REPORT}"
    )

    # ========================================================
    # APPLY NEUTRAL FALLBACK
    #
    # Only genuinely unmapped values reach here.
    # ========================================================

    print(
        "\nApplying neutral WOE fallback..."
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
    # IV REPORT
    # ========================================================

    iv_report = pd.DataFrame(
        iv_results
    )

    iv_report = (
        iv_report
        .sort_values(
            "iv",
            ascending=False
        )
    )

    iv_report.to_csv(
        IV_REPORT,
        index=False
    )

    print(
        "\nIV report created."
    )

    # ========================================================
    # SAVE COMPLETE WOE DATASETS
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

    print(
        "Complete WOE datasets saved."
    )

    # ========================================================
    # FINAL WOE NaN CHECK
    # ========================================================

    woe_features = [
        feature
        for feature in features
        if feature in train_woe.columns
    ]

    train_nan_count = int(
        train_woe[woe_features]
        .isna()
        .sum()
        .sum()
    )

    validation_nan_count = int(
        validation_woe[woe_features]
        .isna()
        .sum()
        .sum()
    )

    test_nan_count = int(
        test_woe[woe_features]
        .isna()
        .sum()
        .sum()
    )

    print(
        "\nWOE NaN check:"
    )

    print(
        f"Train      : "
        f"{train_nan_count:,}"
    )

    print(
        f"Validation : "
        f"{validation_nan_count:,}"
    )

    print(
        f"Test       : "
        f"{test_nan_count:,}"
    )

    if (
        train_nan_count == 0
        and validation_nan_count == 0
        and test_nan_count == 0
    ):

        print(
            "PASS: No NaN values "
            "in WOE features."
        )

    else:

        print(
            "WARNING: Some WOE values "
            "remain NaN."
        )

    # ========================================================
    # IV FEATURE SELECTION
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
    # SAVE SELECTED FEATURES
    # ========================================================

    selected_report = (
        iv_report[
            iv_report["feature"]
            .isin(selected_features)
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
    # SAVE REMOVED FEATURES
    # ========================================================

    removed_report = (
        iv_report[
            iv_report["feature"]
            .isin(removed_features)
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
    # CREATE FINAL MODEL-READY DATASETS
    # ========================================================

    final_columns = (
        ["SK_ID_CURR", "TARGET"]
        + selected_features
    )

    train_final = (
        train_woe[
            final_columns
        ].copy()
    )

    validation_final = (
        validation_woe[
            final_columns
        ].copy()
    )

    test_final = (
        test_woe[
            final_columns
        ].copy()
    )

    # ========================================================
    # SAVE FINAL DATASETS
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
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n============================================================"
    )

    print(
        "TOP 20 FEATURES BY IV"
    )

    print(
        "============================================================"
    )

    print(
        iv_report
        .head(20)
        .to_string(index=False)
    )

    print(
        "\n============================================================"
    )

    print(
        "IV FEATURE SELECTION"
    )

    print(
        "============================================================"
    )

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

    print(
        "\nBinning summary:"
    )

    print(
        iv_report[
            "binning_type"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nFinal model-ready datasets:"
    )

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

    print(
        "\nSaved files:"
    )

    print(TRAIN_OUTPUT)
    print(VAL_OUTPUT)
    print(TEST_OUTPUT)
    print(IV_REPORT)
    print(SELECTED_FEATURES_REPORT)
    print(REMOVED_FEATURES_REPORT)
    print(FALLBACK_REPORT)
    print(TRAIN_FINAL_OUTPUT)
    print(VAL_FINAL_OUTPUT)
    print(TEST_FINAL_OUTPUT)

    print(
        "\nCreditIQ WOE/IV pipeline "
        "completed successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_woe()