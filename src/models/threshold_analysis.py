"""
CreditIQ — Decision Threshold Analysis

Purpose
-------
Analyze decision thresholds using the SAME calibration procedure
already implemented in calibration.py.

Important:
- Model is already trained.
- Platt calibrator is already trained.
- Test data remains completely untouched.
- Threshold analysis uses VALIDATION data only.
- Accuracy is not used as the primary selection metric.

Calibration pipeline:
    XGBoost probability
        ↓
    log-odds transformation
        ↓
    existing Platt calibrator
        ↓
    calibrated probability
        ↓
    threshold analysis
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import train_test_split


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

METRICS_DIR = (
    PROJECT_ROOT
    / "reports"
    / "metrics"
)

VALIDATION_FILE = (
    PROCESSED_DIR
    / "validation_final.parquet"
)

MODEL_FILE = (
    MODEL_DIR
    / "xgboost_monotonic.joblib"
)

CALIBRATOR_FILE = (
    MODEL_DIR
    / "platt_calibrator.joblib"
)

OUTPUT_FILE = (
    METRICS_DIR
    / "threshold_analysis.json"
)


# ============================================================
# CONSTANTS
# ============================================================

TARGET = "TARGET"
ID_COLUMN = "SK_ID_CURR"

RANDOM_STATE = 42

# Analyze thresholds from 0.01 to 0.50.
# We are NOT assuming 0.50 is correct.
THRESHOLDS = np.round(
    np.arange(
        0.01,
        0.51,
        0.01
    ),
    2
)


# ============================================================
# KS STATISTIC
# ============================================================

def calculate_ks(
    y_true,
    probabilities
):
    """
    Calculate the overall KS statistic.
    """

    fpr, tpr, thresholds = roc_curve(
        y_true,
        probabilities
    )

    ks_values = (
        tpr - fpr
    )

    return float(
        np.max(ks_values)
    )


# ============================================================
# PLATT INPUT TRANSFORMATION
# ============================================================

def probability_to_log_odds(
    probabilities
):
    """
    Convert XGBoost probabilities into the exact
    log-odds representation expected by the saved
    Platt calibrator.

    This matches calibration.py.
    """

    clipped = np.clip(
        probabilities,
        1e-6,
        1 - 1e-6
    )

    log_odds = np.log(
        clipped
        /
        (1 - clipped)
    )

    return log_odds.reshape(
        -1,
        1
    )


# ============================================================
# MAIN
# ============================================================

def run_threshold_analysis():

    print(
        "Starting CreditIQ threshold analysis..."
    )

    # ========================================================
    # 1. LOAD MODEL
    # ========================================================

    print(
        "Loading monotonic XGBoost model..."
    )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_FILE}"
        )

    model = joblib.load(
        MODEL_FILE
    )

    # ========================================================
    # 2. LOAD PLATT CALIBRATOR
    # ========================================================

    print(
        "Loading Platt calibrator..."
    )

    if not CALIBRATOR_FILE.exists():
        raise FileNotFoundError(
            f"Platt calibrator not found:\n"
            f"{CALIBRATOR_FILE}"
        )

    calibrator = joblib.load(
        CALIBRATOR_FILE
    )

    # ========================================================
    # 3. LOAD VALIDATION DATA
    # ========================================================

    print(
        "Reading validation dataset..."
    )

    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            f"Validation file not found:\n"
            f"{VALIDATION_FILE}"
        )

    validation = pd.read_parquet(
        VALIDATION_FILE
    )

    print(
        f"Validation: "
        f"{len(validation):,} rows × "
        f"{len(validation.columns):,} columns"
    )

    # ========================================================
    # 4. PREPARE FEATURES
    # ========================================================

    feature_columns = [
        column
        for column in validation.columns
        if column not in [
            TARGET,
            ID_COLUMN
        ]
    ]

    X = validation[
        feature_columns
    ]

    y = validation[
        TARGET
    ]

    print(
        f"Validation features: "
        f"{len(feature_columns)}"
    )

    # ========================================================
    # 5. RECREATE THE SAME VALIDATION SPLIT
    #
    # This MUST match calibration.py:
    #
    # test_size=0.50
    # random_state=42
    # stratify=y
    # ========================================================

    print(
        "\nRecreating calibration/evaluation split..."
    )

    (
        X_calibration,
        X_evaluation,
        y_calibration,
        y_evaluation
    ) = train_test_split(
        X,
        y,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(
        f"Calibration set: "
        f"{len(X_calibration):,}"
    )

    print(
        f"Evaluation set: "
        f"{len(X_evaluation):,}"
    )

    # ========================================================
    # 6. GENERATE XGBOOST PROBABILITIES
    #
    # We do this on the evaluation portion because the
    # existing Platt calibrator was fitted using the
    # calibration portion.
    # ========================================================

    print(
        "\nGenerating XGBoost probabilities..."
    )

    evaluation_probabilities = (
        model.predict_proba(
            X_evaluation
        )[:, 1]
    )

    # ========================================================
    # 7. CONVERT PROBABILITY → LOG-ODDS
    #
    # IMPORTANT:
    # This is the correction to the previous script.
    #
    # calibration.py does:
    #
    # probability
    #      ↓
    # log(p / (1-p))
    #      ↓
    # Platt calibrator
    # ========================================================

    print(
        "Converting probabilities to log-odds..."
    )

    evaluation_scores = (
        probability_to_log_odds(
            evaluation_probabilities
        )
    )

    # ========================================================
    # 8. APPLY EXISTING PLATT CALIBRATOR
    # ========================================================

    print(
        "Applying Platt calibration..."
    )

    calibrated_probabilities = (
        calibrator.predict_proba(
            evaluation_scores
        )[:, 1]
    )

    # ========================================================
    # 9. THRESHOLD-INDEPENDENT METRICS
    # ========================================================

    roc_auc = roc_auc_score(
        y_evaluation,
        calibrated_probabilities
    )

    pr_auc = average_precision_score(
        y_evaluation,
        calibrated_probabilities
    )

    ks = calculate_ks(
        y_evaluation,
        calibrated_probabilities
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "THRESHOLD-INDEPENDENT METRICS"
    )

    print(
        "=" * 60
    )

    print(
        f"ROC_AUC : {roc_auc:.4f}"
    )

    print(
        f"PR_AUC  : {pr_auc:.4f}"
    )

    print(
        f"KS      : {ks:.4f}"
    )

    # ========================================================
    # 10. THRESHOLD ANALYSIS
    # ========================================================

    threshold_results = []

    print(
        "\n"
        + "=" * 60
    )

    print(
        "THRESHOLD ANALYSIS"
    )

    print(
        "=" * 60
    )

    for threshold in THRESHOLDS:

        predictions = (
            calibrated_probabilities
            >= threshold
        ).astype(int)

        (
            tn,
            fp,
            fn,
            tp
        ) = confusion_matrix(
            y_evaluation,
            predictions,
            labels=[0, 1]
        ).ravel()

        accuracy = accuracy_score(
            y_evaluation,
            predictions
        )

        precision = precision_score(
            y_evaluation,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_evaluation,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_evaluation,
            predictions,
            zero_division=0
        )

        false_positive_rate = (
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0.0
        )

        true_positive_rate = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        threshold_ks = (
            true_positive_rate
            - false_positive_rate
        )

        approval_rate = (
            (predictions == 0).mean()
        )

        decline_rate = (
            (predictions == 1).mean()
        )

        result = {

            "threshold": float(
                threshold
            ),

            "accuracy": float(
                accuracy
            ),

            "precision": float(
                precision
            ),

            "recall": float(
                recall
            ),

            "f1": float(
                f1
            ),

            "false_positive_rate": float(
                false_positive_rate
            ),

            "true_positive_rate": float(
                true_positive_rate
            ),

            "threshold_ks": float(
                threshold_ks
            ),

            "approval_rate": float(
                approval_rate
            ),

            "decline_rate": float(
                decline_rate
            ),

            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }

        threshold_results.append(
            result
        )

    # ========================================================
    # 11. REFERENCE THRESHOLDS
    # ========================================================

    best_f1_result = max(
        threshold_results,
        key=lambda x: x["f1"]
    )

    best_ks_result = max(
        threshold_results,
        key=lambda x: x["threshold_ks"]
    )

    # ========================================================
    # 12. DISPLAY RESULTS
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "BEST REFERENCE THRESHOLDS"
    )

    print(
        "=" * 60
    )

    print(
        "\nMaximum F1 threshold:"
    )

    print(
        f"Threshold : "
        f"{best_f1_result['threshold']:.2f}"
    )

    print(
        f"Precision : "
        f"{best_f1_result['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{best_f1_result['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{best_f1_result['f1']:.4f}"
    )

    print(
        f"FPR       : "
        f"{best_f1_result['false_positive_rate']:.4f}"
    )

    print(
        f"Approval  : "
        f"{best_f1_result['approval_rate']:.4f}"
    )

    print(
        "\nMaximum threshold-level KS separation:"
    )

    print(
        f"Threshold : "
        f"{best_ks_result['threshold']:.2f}"
    )

    print(
        f"Threshold KS : "
        f"{best_ks_result['threshold_ks']:.4f}"
    )

    print(
        f"Precision    : "
        f"{best_ks_result['precision']:.4f}"
    )

    print(
        f"Recall       : "
        f"{best_ks_result['recall']:.4f}"
    )

    print(
        f"FPR          : "
        f"{best_ks_result['false_positive_rate']:.4f}"
    )

    print(
        f"Approval     : "
        f"{best_ks_result['approval_rate']:.4f}"
    )

    # ========================================================
    # 13. THRESHOLD TABLE
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "THRESHOLD TABLE"
    )

    print(
        "=" * 60
    )

    print(
        f"{'Threshold':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
        f"{'FPR':<12}"
        f"{'Approval':<12}"
    )

    print(
        "-" * 72
    )

    for result in threshold_results:

        print(
            f"{result['threshold']:<12.2f}"
            f"{result['precision']:<12.4f}"
            f"{result['recall']:<12.4f}"
            f"{result['f1']:<12.4f}"
            f"{result['false_positive_rate']:<12.4f}"
            f"{result['approval_rate']:<12.4f}"
        )

    # ========================================================
    # 14. SAVE REPORT
    # ========================================================

    output = {

        "model":
            "monotonic_xgboost",

        "calibration":
            "existing_platt_calibrator",

        "calibration_input":
            "xgboost_probability_log_odds",

        "validation_rows":
            int(len(validation)),

        "calibration_rows":
            int(len(X_calibration)),

        "evaluation_rows":
            int(len(X_evaluation)),

        "roc_auc":
            float(roc_auc),

        "pr_auc":
            float(pr_auc),

        "overall_ks":
            float(ks),

        "maximum_f1_reference":
            best_f1_result,

        "maximum_threshold_ks_reference":
            best_ks_result,

        "thresholds":
            threshold_results,

        "note":
            (
                "Threshold analysis was performed on "
                "the validation evaluation split only. "
                "The test dataset was not accessed. "
                "Reference thresholds are reported for "
                "policy selection and are not automatically "
                "treated as the final production threshold."
            )
    }

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4
        )

    # ========================================================
    # 15. FINAL MESSAGE
    # ========================================================

    print(
        "\nSaved threshold analysis to:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nCreditIQ threshold analysis completed successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_threshold_analysis()