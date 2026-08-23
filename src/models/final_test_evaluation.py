"""
CreditIQ — Final Test Evaluation

Evaluates the finalized monotonic XGBoost + Platt calibration
pipeline once on the untouched test dataset.

IMPORTANT
---------
- The model was trained before this stage.
- The Platt calibrator was trained before this stage.
- The decision threshold was selected using VALIDATION data.
- The threshold is frozen in configs/threshold.yaml.
- TEST data is used only for final evaluation.
- Accuracy is reported but is NOT the headline metric.

Final pipeline:

    Untouched Test Data
            ↓
    Monotonic XGBoost
            ↓
    Raw probability
            ↓
    Log-odds transformation
            ↓
    Platt calibration
            ↓
    Frozen decision threshold
            ↓
    Final decision + evaluation metrics
"""


from pathlib import Path
import json
import re

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
    classification_report,
    roc_curve,
)
from sklearn.calibration import calibration_curve


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)


DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_final.parquet"
)


MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "xgboost_monotonic.joblib"
)


CALIBRATOR_FILE = (
    PROJECT_ROOT
    / "models"
    / "platt_calibrator.joblib"
)


THRESHOLD_FILE = (
    PROJECT_ROOT
    / "configs"
    / "threshold.yaml"
)


METRICS_DIR = (
    PROJECT_ROOT
    / "reports"
    / "metrics"
)


METRICS_FILE = (
    METRICS_DIR
    / "final_test_results.json"
)


# ============================================================
# CONSTANTS
# ============================================================

TARGET = "TARGET"

ID_COLUMN = "SK_ID_CURR"


# ============================================================
# LOAD FROZEN THRESHOLD
# ============================================================

def load_decision_threshold():

    """
    Load the frozen decision threshold from:

        configs/threshold.yaml

    We intentionally parse only the decision_threshold
    value rather than introducing an unnecessary dependency
    on a YAML package.
    """

    print(
        "\nLoading frozen decision threshold..."
    )

    if not THRESHOLD_FILE.exists():

        raise FileNotFoundError(
            "Decision threshold configuration "
            "was not found:\n"
            f"{THRESHOLD_FILE}"
        )

    text = THRESHOLD_FILE.read_text(
        encoding="utf-8"
    )

    match = re.search(
        r"^\s*decision_threshold\s*:\s*"
        r"([0-9]*\.?[0-9]+)\s*$",
        text,
        flags=re.MULTILINE
    )

    if match is None:

        raise ValueError(
            "Could not find 'decision_threshold' "
            "in threshold.yaml."
        )

    threshold = float(
        match.group(1)
    )

    if not 0 < threshold < 1:

        raise ValueError(
            "Decision threshold must be "
            "between 0 and 1."
        )

    print(
        f"Frozen decision threshold: "
        f"{threshold:.4f}"
    )

    return threshold


# ============================================================
# KS STATISTIC
# ============================================================

def calculate_ks(
    y_true,
    probabilities
):

    """
    Calculate the KS statistic using the
    calibrated probability scores.
    """

    fpr, tpr, _ = roc_curve(
        y_true,
        probabilities
    )

    return float(
        np.max(
            tpr - fpr
        )
    )


# ============================================================
# CALIBRATION ERROR
# ============================================================

def calculate_calibration_error(
    y_true,
    probabilities
):

    """
    Calculate mean absolute calibration error
    using 10 quantile bins.
    """

    observed, predicted = (
        calibration_curve(
            y_true,
            probabilities,
            n_bins=10,
            strategy="quantile"
        )
    )

    return float(
        np.mean(
            np.abs(
                observed - predicted
            )
        )
    )


# ============================================================
# MAIN
# ============================================================

def run_final_test_evaluation():

    print(
        "Starting CreditIQ final test evaluation..."
    )

    # ========================================================
    # 1. Load frozen threshold FIRST
    # ========================================================

    threshold = (
        load_decision_threshold()
    )

    # ========================================================
    # 2. Load untouched test data
    # ========================================================

    print(
        "\nReading untouched test dataset..."
    )

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Test dataset not found:\n"
            f"{DATA_FILE}"
        )

    test = pd.read_parquet(
        DATA_FILE
    )

    print(
        f"Test: "
        f"{len(test):,} rows × "
        f"{len(test.columns):,} columns"
    )

    # ========================================================
    # 3. Load finalized model
    # ========================================================

    print(
        "\nLoading monotonic XGBoost..."
    )

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"Model not found:\n"
            f"{MODEL_FILE}"
        )

    model = joblib.load(
        MODEL_FILE
    )

    # ========================================================
    # 4. Load existing Platt calibrator
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
    # 5. Prepare test features
    # ========================================================

    feature_columns = [
        column
        for column in test.columns
        if column not in [
            TARGET,
            ID_COLUMN
        ]
    ]

    X_test = test[
        feature_columns
    ]

    y_test = test[
        TARGET
    ]

    print(
        f"Test features: "
        f"{len(feature_columns)}"
    )

    # ========================================================
    # 6. Generate XGBoost probabilities
    # ========================================================

    print(
        "\nGenerating XGBoost test probabilities..."
    )

    raw_probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    # ========================================================
    # 7. Convert probability → log-odds
    #
    # This matches the existing calibration.py pipeline.
    # ========================================================

    print(
        "Converting probabilities to log-odds..."
    )

    clipped_probabilities = np.clip(
        raw_probabilities,
        1e-6,
        1 - 1e-6
    )

    raw_scores = (
        np.log(
            clipped_probabilities
            /
            (
                1
                -
                clipped_probabilities
            )
        )
        .reshape(-1, 1)
    )

    # ========================================================
    # 8. Apply Platt calibration
    # ========================================================

    print(
        "Applying Platt calibration..."
    )

    calibrated_probabilities = (
        calibrator.predict_proba(
            raw_scores
        )[:, 1]
    )

    # ========================================================
    # 9. Apply FROZEN decision threshold
    #
    # IMPORTANT:
    # This threshold came from VALIDATION.
    # It is NOT selected using TEST.
    # ========================================================

    print(
        "\nApplying frozen decision threshold..."
    )

    print(
        f"Threshold: {threshold:.4f}"
    )

    predictions = (
        calibrated_probabilities
        >= threshold
    ).astype(int)

    # ========================================================
    # 10. Calculate confusion matrix
    # ========================================================

    (
        tn,
        fp,
        fn,
        tp
    ) = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    # ========================================================
    # 11. Calculate threshold-based metrics
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
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
        -
        false_positive_rate
    )

    approval_rate = (
        (predictions == 0).mean()
    )

    decline_rate = (
        (predictions == 1).mean()
    )

    # ========================================================
    # 12. Calculate threshold-independent metrics
    # ========================================================

    roc_auc = roc_auc_score(
        y_test,
        calibrated_probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        calibrated_probabilities
    )

    ks = calculate_ks(
        y_test,
        calibrated_probabilities
    )

    calibration_error = (
        calculate_calibration_error(
            y_test,
            calibrated_probabilities
        )
    )

    # ========================================================
    # 13. Create final metrics dictionary
    # ========================================================

    metrics = {

        # Threshold-independent
        "roc_auc": float(
            roc_auc
        ),

        "pr_auc": float(
            pr_auc
        ),

        "ks": float(
            ks
        ),

        "calibration_error": float(
            calibration_error
        ),

        # Threshold-dependent
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

        # Confusion matrix counts
        "true_negatives": int(
            tn
        ),

        "false_positives": int(
            fp
        ),

        "false_negatives": int(
            fn
        ),

        "true_positives": int(
            tp
        )
    }

    # ========================================================
    # 14. Display final results
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "FINAL TEST RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        "\nThreshold-independent metrics:"
    )

    print(
        f"ROC_AUC             : "
        f"{roc_auc:.4f}"
    )

    print(
        f"PR_AUC              : "
        f"{pr_auc:.4f}"
    )

    print(
        f"KS                  : "
        f"{ks:.4f}"
    )

    print(
        f"CALIBRATION_ERROR   : "
        f"{calibration_error:.4f}"
    )

    print(
        "\nThreshold-dependent metrics:"
    )

    print(
        f"THRESHOLD           : "
        f"{threshold:.4f}"
    )

    print(
        f"ACCURACY            : "
        f"{accuracy:.4f}"
    )

    print(
        f"PRECISION           : "
        f"{precision:.4f}"
    )

    print(
        f"RECALL              : "
        f"{recall:.4f}"
    )

    print(
        f"F1                  : "
        f"{f1:.4f}"
    )

    print(
        f"FALSE_POSITIVE_RATE : "
        f"{false_positive_rate:.4f}"
    )

    print(
        f"TRUE_POSITIVE_RATE  : "
        f"{true_positive_rate:.4f}"
    )

    print(
        f"THRESHOLD_KS        : "
        f"{threshold_ks:.4f}"
    )

    print(
        f"APPROVAL_RATE       : "
        f"{approval_rate:.4f}"
    )

    print(
        f"DECLINE_RATE        : "
        f"{decline_rate:.4f}"
    )

    # ========================================================
    # 15. Confusion matrix
    # ========================================================

    print(
        "\nConfusion Matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    # ========================================================
    # 16. Classification report
    # ========================================================

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # ========================================================
    # 17. Save final test results
    # ========================================================

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results = {

        "model":
            "Monotonic XGBoost",

        "calibration":
            "Platt scaling",

        "threshold_source":
            "configs/threshold.yaml",

        "threshold_selection_data":
            "validation",

        "test_data_usage":
            "final evaluation only",

        "test_rows":
            int(len(test)),

        "test_features":
            int(len(feature_columns)),

        "threshold":
            float(threshold),

        "metrics":
            metrics
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    # ========================================================
    # 18. Completion message
    # ========================================================

    print(
        "\nSaved final test results to:"
    )

    print(
        METRICS_FILE
    )

    print(
        "\nCreditIQ final test evaluation "
        "completed successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_final_test_evaluation()