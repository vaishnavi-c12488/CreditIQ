"""
CreditIQ — Fairness Audit using Fairlearn

Creates an age-based proxy from the original DAYS_BIRTH values,
evaluates fairness metrics, applies fairness mitigation, and
compares fairness and decision performance before and after mitigation.

The audit uses the finalized Monotonic XGBoost model,
Platt calibration, and the frozen decision threshold.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    selection_rate,
)

from fairlearn.postprocessing import ThresholdOptimizer


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

RAW_VALIDATION_FILE = (
    PROCESSED_DIR
    / "validation.parquet"
)

MODEL_VALIDATION_FILE = (
    PROCESSED_DIR
    / "validation_final.parquet"
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

REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
    / "metrics"
)

REPORT_FILE = (
    REPORTS_DIR
    / "fairness_audit.json"
)


# ============================================================
# Decision policy
# ============================================================

DECISION_THRESHOLD = 0.20


# ============================================================
# Age proxy configuration
# ============================================================

AGE_BINS = [
    0,
    25,
    35,
    45,
    55,
    np.inf,
]

AGE_LABELS = [
    "<25",
    "25-34",
    "35-44",
    "45-54",
    "55+",
]


# ============================================================
# Helper functions
# ============================================================

def create_age_groups(days_birth):
    """
    Convert original DAYS_BIRTH values into age brackets.
    """

    age_years = (
        days_birth.abs()
        / 365.25
    )

    age_groups = pd.cut(
        age_years,
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=False,
    )

    return age_groups.astype(str)


def calculate_fairness_metrics(
    y_true,
    predictions,
    sensitive_features,
):
    """
    Calculate fairness metrics.
    """

    dp_difference = (
        demographic_parity_difference(
            y_true=y_true,
            y_pred=predictions,
            sensitive_features=sensitive_features,
        )
    )

    eo_difference = (
        equalized_odds_difference(
            y_true=y_true,
            y_pred=predictions,
            sensitive_features=sensitive_features,
        )
    )

    return {
        "demographic_parity_difference": float(
            dp_difference
        ),
        "equalized_odds_difference": float(
            eo_difference
        ),
    }


def calculate_decision_metrics(
    y_true_default,
    default_predictions,
):
    """
    Calculate threshold-dependent model performance.

    TARGET:
        1 = default
        0 = non-default
    """

    return {
        "accuracy": float(
            accuracy_score(
                y_true_default,
                default_predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_true_default,
                default_predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true_default,
                default_predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true_default,
                default_predictions,
                zero_division=0,
            )
        ),
    }


def calculate_operating_metrics(
    y_true_default,
    default_predictions,
):
    """
    Calculate credit-decision operating metrics.
    """

    tn, fp, fn, tp = confusion_matrix(
        y_true_default,
        default_predictions,
        labels=[0, 1],
    ).ravel()

    total = (
        tn + fp + fn + tp
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    approval_predictions = (
        1 - default_predictions
    )

    approval_rate = (
        approval_predictions.mean()
    )

    decline_rate = (
        1 - approval_rate
    )

    return {
        "false_positive_rate": float(
            false_positive_rate
        ),
        "true_positive_rate": float(
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        ),
        "approval_rate": float(
            approval_rate
        ),
        "decline_rate": float(
            decline_rate
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "total": int(total),
    }


def print_selection_rates(
    predictions,
    sensitive_features,
    title,
):
    """
    Print approval/selection rate by age group.
    """

    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)

    rates = {}

    for group in AGE_LABELS:

        mask = (
            sensitive_features == group
        )

        if mask.sum() == 0:
            continue

        rate = selection_rate(
            y_true=None,
            y_pred=predictions[mask],
        )

        rates[group] = float(rate)

        print(
            f"{group:>8} : "
            f"{rate:.4f} "
            f"({rate * 100:.2f}%)"
        )

    return rates


def print_performance(
    title,
    decision_metrics,
    operating_metrics,
):
    """
    Display threshold-dependent performance.
    """

    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)

    print(
        f"Accuracy            : "
        f"{decision_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision           : "
        f"{decision_metrics['precision']:.4f}"
    )

    print(
        f"Recall              : "
        f"{decision_metrics['recall']:.4f}"
    )

    print(
        f"F1                  : "
        f"{decision_metrics['f1']:.4f}"
    )

    print(
        f"False Positive Rate : "
        f"{operating_metrics['false_positive_rate']:.4f}"
    )

    print(
        f"Approval Rate       : "
        f"{operating_metrics['approval_rate']:.4f}"
    )

    print(
        f"Decline Rate        : "
        f"{operating_metrics['decline_rate']:.4f}"
    )


# ============================================================
# Sklearn-compatible approval estimator
# ============================================================

class ApprovalProbabilityEstimator(
    BaseEstimator,
    ClassifierMixin,
):
    """
    Adapter which converts default probability into
    approval probability.

    TARGET:
        1 = default

    Approval:
        1 = approve
        0 = decline
    """

    def __init__(self, fitted_model):
        self.fitted_model = fitted_model

    def fit(self, X, y):
        self.is_fitted_ = True
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):

        default_probability = (
            self.fitted_model
            .predict_proba(X)[:, 1]
        )

        approval_probability = (
            1 - default_probability
        )

        return np.column_stack(
            [
                1 - approval_probability,
                approval_probability,
            ]
        )

    def predict(self, X):

        approval_probability = (
            self.predict_proba(X)[:, 1]
        )

        return (
            approval_probability >= 0.5
        ).astype(int)


# ============================================================
# Main
# ============================================================

def run_fairness_audit():

    print("=" * 70)
    print("CreditIQ — FAIRNESS AUDIT")
    print("=" * 70)

    # ========================================================
    # 1. Load original validation data
    # ========================================================

    print(
        "\nLoading original validation data..."
    )

    raw_validation = pd.read_parquet(
        RAW_VALIDATION_FILE
    )

    print(
        f"Original validation: "
        f"{len(raw_validation):,} rows × "
        f"{len(raw_validation.columns):,} columns"
    )

    # ========================================================
    # 2. Load model-ready validation data
    # ========================================================

    print(
        "\nLoading model-ready validation data..."
    )

    validation = pd.read_parquet(
        MODEL_VALIDATION_FILE
    )

    print(
        f"Model validation: "
        f"{len(validation):,} rows × "
        f"{len(validation.columns):,} columns"
    )

    # ========================================================
    # 3. Validate alignment
    # ========================================================

    required_columns = {
        "DAYS_BIRTH",
        "TARGET",
    }

    missing = (
        required_columns
        - set(raw_validation.columns)
    )

    if missing:
        raise ValueError(
            f"Missing columns in validation.parquet: "
            f"{missing}"
        )

    if len(raw_validation) != len(validation):
        raise ValueError(
            "Raw validation and model validation "
            "have different row counts."
        )

    if not np.array_equal(
        raw_validation["TARGET"].to_numpy(),
        validation["TARGET"].to_numpy(),
    ):
        raise ValueError(
            "TARGET ordering differs between "
            "validation datasets."
        )

    print(
        "\nPASS: Raw and model validation rows "
        "are aligned."
    )

    # ========================================================
    # 4. Load model
    # ========================================================

    print(
        "\nLoading finalized Monotonic XGBoost..."
    )

    model = joblib.load(
        MODEL_FILE
    )

    print(
        f"Model loaded from:\n"
        f"  {MODEL_FILE}"
    )

    # ========================================================
    # 5. Load calibrator
    # ========================================================

    print(
        "\nLoading Platt calibrator..."
    )

    calibrator = joblib.load(
        CALIBRATOR_FILE
    )

    # ========================================================
    # 6. Prepare model features
    # ========================================================

    TARGET = "TARGET"
    ID_COLUMN = "SK_ID_CURR"

    feature_columns = [
        column
        for column in validation.columns
        if column not in [
            TARGET,
            ID_COLUMN,
        ]
    ]

    X_validation = validation[
        feature_columns
    ].copy()

    y_default = (
        validation[TARGET]
        .astype(int)
    )

    print(
        f"Model features: "
        f"{len(feature_columns)}"
    )

    # ========================================================
    # 7. Create age proxy
    # ========================================================

    print("\n" + "=" * 70)
    print("CREATING AGE PROXY")
    print("=" * 70)

    age_groups = create_age_groups(
        raw_validation["DAYS_BIRTH"]
    )

    group_counts = (
        age_groups
        .value_counts()
        .reindex(AGE_LABELS)
        .fillna(0)
        .astype(int)
    )

    print("\nAge group distribution:")

    for group, count in group_counts.items():

        print(
            f"  {group:>8}: "
            f"{count:,}"
        )

    if age_groups.nunique() < 2:
        raise ValueError(
            "At least two age groups are required."
        )

    # ========================================================
    # 8. Generate calibrated probabilities
    # ========================================================

    print("\n" + "=" * 70)
    print("GENERATING CALIBRATED PROBABILITIES")
    print("=" * 70)

    raw_default_probability = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    clipped_probability = np.clip(
        raw_default_probability,
        1e-6,
        1 - 1e-6,
    )

    raw_log_odds = (
        np.log(
            clipped_probability
            /
            (1 - clipped_probability)
        )
    ).reshape(-1, 1)

    calibrated_default_probability = (
        calibrator.predict_proba(
            raw_log_odds
        )[:, 1]
    )

    print(
        "Platt calibration applied."
    )

    # ========================================================
    # 9. Threshold-independent model metrics
    # ========================================================

    roc_auc = roc_auc_score(
        y_default,
        calibrated_default_probability,
    )

    pr_auc = average_precision_score(
        y_default,
        calibrated_default_probability,
    )

    print("\n" + "=" * 70)
    print("THRESHOLD-INDEPENDENT MODEL METRICS")
    print("=" * 70)

    print(
        f"ROC_AUC : {roc_auc:.4f}"
    )

    print(
        f"PR_AUC  : {pr_auc:.4f}"
    )

    # ========================================================
    # 10. Frozen decision threshold
    # ========================================================

    print("\n" + "=" * 70)
    print("APPLYING FROZEN DECISION THRESHOLD")
    print("=" * 70)

    print(
        f"Threshold: "
        f"{DECISION_THRESHOLD:.4f}"
    )

    before_default_predictions = (
        calibrated_default_probability
        >= DECISION_THRESHOLD
    ).astype(int)

    before_approval_predictions = (
        1 - before_default_predictions
    )

    # Actual successful outcome:
    #
    # 1 = non-default
    # 0 = default

    actual_success = (
        1 - y_default
    )

    # ========================================================
    # 11. BEFORE MITIGATION
    # ========================================================

    print("\n" + "=" * 70)
    print("BEFORE MITIGATION")
    print("=" * 70)

    before_fairness = (
        calculate_fairness_metrics(
            actual_success,
            before_approval_predictions,
            age_groups,
        )
    )

    before_decision = (
        calculate_decision_metrics(
            y_default,
            before_default_predictions,
        )
    )

    before_operating = (
        calculate_operating_metrics(
            y_default,
            before_default_predictions,
        )
    )

    print_performance(
        "Decision Performance",
        before_decision,
        before_operating,
    )

    before_selection_rates = (
        print_selection_rates(
            before_approval_predictions,
            age_groups,
            "Approval Rate by Age Group",
        )
    )

    print(
        "\nFairness:"
    )

    print(
        f"Demographic Parity Difference : "
        f"{before_fairness['demographic_parity_difference']:.6f}"
    )

    print(
        f"Equalized Odds Difference     : "
        f"{before_fairness['equalized_odds_difference']:.6f}"
    )

    # ========================================================
    # 12. Fairness mitigation
    # ========================================================

    print("\n" + "=" * 70)
    print("FAIRNESS MITIGATION")
    print("=" * 70)

    approval_estimator = (
        ApprovalProbabilityEstimator(
            fitted_model=model
        )
    )

    # Mark the adapter as fitted without retraining
    approval_estimator.fit(
        X_validation,
        actual_success,
    )

    threshold_optimizer = (
        ThresholdOptimizer(
            estimator=approval_estimator,
            constraints="equalized_odds",
            objective="balanced_accuracy_score",
            predict_method="predict_proba",
            prefit=True,
        )
    )

    print(
        "Fitting Fairlearn ThresholdOptimizer..."
    )

    threshold_optimizer.fit(
        X_validation,
        actual_success,
        sensitive_features=age_groups,
    )

    mitigated_approval_predictions = (
        threshold_optimizer.predict(
            X_validation,
            sensitive_features=age_groups,
        ).astype(int)
    )

    # ========================================================
    # 13. Convert mitigated approval decision
    # ========================================================

    mitigated_default_predictions = (
        1 - mitigated_approval_predictions
    )

    # ========================================================
    # 14. AFTER MITIGATION
    # ========================================================

    print("\n" + "=" * 70)
    print("AFTER MITIGATION")
    print("=" * 70)

    after_fairness = (
        calculate_fairness_metrics(
            actual_success,
            mitigated_approval_predictions,
            age_groups,
        )
    )

    after_decision = (
        calculate_decision_metrics(
            y_default,
            mitigated_default_predictions,
        )
    )

    after_operating = (
        calculate_operating_metrics(
            y_default,
            mitigated_default_predictions,
        )
    )

    print_performance(
        "Decision Performance",
        after_decision,
        after_operating,
    )

    after_selection_rates = (
        print_selection_rates(
            mitigated_approval_predictions,
            age_groups,
            "Approval Rate by Age Group",
        )
    )

    print(
        "\nFairness:"
    )

    print(
        f"Demographic Parity Difference : "
        f"{after_fairness['demographic_parity_difference']:.6f}"
    )

    print(
        f"Equalized Odds Difference     : "
        f"{after_fairness['equalized_odds_difference']:.6f}"
    )

    # ========================================================
    # 15. Comparison
    # ========================================================

    print("\n" + "=" * 70)
    print("BEFORE vs AFTER MITIGATION")
    print("=" * 70)

    print(
        f"\n{'Metric':<35}"
        f"{'Before':>15}"
        f"{'After':>15}"
        f"{'Change':>15}"
    )

    print("-" * 80)

    comparisons = [
        (
            "Accuracy",
            before_decision["accuracy"],
            after_decision["accuracy"],
        ),
        (
            "Precision",
            before_decision["precision"],
            after_decision["precision"],
        ),
        (
            "Recall",
            before_decision["recall"],
            after_decision["recall"],
        ),
        (
            "F1",
            before_decision["f1"],
            after_decision["f1"],
        ),
        (
            "False Positive Rate",
            before_operating["false_positive_rate"],
            after_operating["false_positive_rate"],
        ),
        (
            "Approval Rate",
            before_operating["approval_rate"],
            after_operating["approval_rate"],
        ),
        (
            "Decline Rate",
            before_operating["decline_rate"],
            after_operating["decline_rate"],
        ),
        (
            "Demographic Parity Difference",
            before_fairness[
                "demographic_parity_difference"
            ],
            after_fairness[
                "demographic_parity_difference"
            ],
        ),
        (
            "Equalized Odds Difference",
            before_fairness[
                "equalized_odds_difference"
            ],
            after_fairness[
                "equalized_odds_difference"
            ],
        ),
    ]

    for name, before, after in comparisons:

        print(
            f"{name:<35}"
            f"{before:>15.6f}"
            f"{after:>15.6f}"
            f"{(after - before):>15.6f}"
        )

    # ========================================================
    # 16. Interpretation
    # ========================================================

    print("\n" + "=" * 70)
    print("FAIRNESS AUDIT INTERPRETATION")
    print("=" * 70)

    eo_before = (
        before_fairness[
            "equalized_odds_difference"
        ]
    )

    eo_after = (
        after_fairness[
            "equalized_odds_difference"
        ]
    )

    dp_before = (
        before_fairness[
            "demographic_parity_difference"
        ]
    )

    dp_after = (
        after_fairness[
            "demographic_parity_difference"
        ]
    )

    if abs(eo_after) < abs(eo_before):

        print(
            "\nPASS: Equalized odds disparity "
            "was reduced."
        )

    else:

        print(
            "\nWARNING: Equalized odds disparity "
            "was not reduced."
        )

    if abs(dp_after) < abs(dp_before):

        print(
            "PASS: Demographic parity disparity "
            "was reduced."
        )

    else:

        print(
            "WARNING: Demographic parity disparity "
            "was not reduced."
        )

    print(
        "\nPrimary fairness metric: "
        "Equalized Odds Difference."
    )

    print(
        "\nImportant limitation:"
    )

    print(
        "DAYS_BIRTH-derived age group is a documented "
        "proxy attribute. This audit does not establish "
        "complete legal or ethical fairness."
    )

    # ========================================================
    # 17. Save report
    # ========================================================

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "model": "Monotonic XGBoost",
        "calibration": "Platt scaling",
        "evaluation_dataset": "validation",
        "test_set_used": False,
        "decision_threshold": DECISION_THRESHOLD,

        "age_proxy": {
            "source": "DAYS_BIRTH",
            "groups": AGE_LABELS,
            "group_counts": {
                str(k): int(v)
                for k, v in group_counts.items()
            },
        },

        "threshold_independent_model_metrics": {
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
        },

        "before_mitigation": {
            "fairness": before_fairness,
            "decision_metrics": before_decision,
            "operating_metrics": before_operating,
            "selection_rates": before_selection_rates,
        },

        "after_mitigation": {
            "fairness": after_fairness,
            "decision_metrics": after_decision,
            "operating_metrics": after_operating,
            "selection_rates": after_selection_rates,
        },

        "mitigation": {
            "method": "Fairlearn ThresholdOptimizer",
            "constraint": "equalized_odds",
            "objective": "balanced_accuracy_score",
        },

        "primary_fairness_metric": (
            "equalized_odds_difference"
        ),

        "limitations": (
            "DAYS_BIRTH-derived age group is a proxy "
            "attribute and does not establish complete "
            "legal or ethical fairness."
        ),
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    print(
        f"\nSaved fairness audit to:"
        f"\n{REPORT_FILE}"
    )

    print("\n" + "=" * 70)
    print(
        "CreditIQ fairness audit completed successfully."
    )
    print("=" * 70)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    run_fairness_audit()