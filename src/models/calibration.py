"""
============================================================
CreditIQ — Model Evaluation & Probability Calibration
============================================================

This module evaluates the monotonic XGBoost model and calibrates
its predicted default probabilities.

WHAT WAS DONE
-------------
1. Monotonic XGBoost was validated using a feature sweep.
   EXT_SOURCE_1, EXT_SOURCE_2 and EXT_SOURCE_3 had 0 violations,
   confirming that the required monotonic constraints are respected.

2. Model performance was evaluated using:
   - ROC-AUC
   - PR-AUC
   - KS-statistic
   - Calibration curve

   Validation results:
   ROC-AUC : 0.7643
   PR-AUC  : 0.2723
   KS      : 0.3921

3. Calibration analysis showed significant miscalibration.
   Calibration error = 0.3429.

4. Because predicted probabilities did not match observed default
   rates well, Platt scaling was applied to correct the probabilities.

5. Validation data is split into:
   - Calibration set → fits the Platt scaler
   - Evaluation set → evaluates calibration improvement

6. Before vs after calibration is compared using:
   - ROC-AUC
   - PR-AUC
   - KS-statistic
   - Calibration error

7. The final TEST set remains untouched.

OUTPUTS
-------
Model:
    models/platt_calibrator.joblib

Metrics:
    reports/metrics/calibration_results.json

Figure:
    reports/figures/calibration_curve_comparison.png

Overall flow:
    Monotonic XGBoost
          ↓
    Evaluate performance
          ↓
    Detect miscalibration
          ↓
    Platt scaling
          ↓
    Re-evaluate
          ↓
    Compare before vs after
============================================================
"""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

MODEL_FILE = MODEL_DIR / "xgboost_monotonic.joblib"

VALIDATION_FILE = (
    PROCESSED_DIR / "validation_final.parquet"
)

CALIBRATOR_FILE = (
    MODEL_DIR / "platt_calibrator.joblib"
)

METRICS_FILE = (
    METRICS_DIR / "calibration_results.json"
)

PLOT_FILE = (
    FIGURES_DIR / "calibration_curve_comparison.png"
)


# ============================================================
# Helper functions
# ============================================================

def calculate_ks(y_true, probabilities):

    fpr, tpr, _ = roc_curve(
        y_true,
        probabilities
    )

    return np.max(tpr - fpr)


def calculate_calibration_error(
    y_true,
    probabilities
):

    observed, predicted = calibration_curve(
        y_true,
        probabilities,
        n_bins=10,
        strategy="quantile"
    )

    return float(
        np.mean(
            np.abs(
                observed - predicted
            )
        )
    )


# ============================================================
# Main calibration pipeline
# ============================================================

def run_calibration():

    print(
        "Starting CreditIQ probability calibration..."
    )

    # ========================================================
    # 1. Load monotonic XGBoost
    # ========================================================

    print(
        "Loading monotonic XGBoost model..."
    )

    model = joblib.load(
        MODEL_FILE
    )

    # ========================================================
    # 2. Load validation data
    # ========================================================

    print(
        "Reading validation dataset..."
    )

    validation = pd.read_parquet(
        VALIDATION_FILE
    )

    TARGET = "TARGET"
    ID_COLUMN = "SK_ID_CURR"

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
        f"Validation: "
        f"{len(validation):,} rows × "
        f"{len(validation.columns):,} columns"
    )

    # ========================================================
    # 3. Split validation data
    # ========================================================

    print(
        "\nSplitting validation data..."
    )

    X_calibration, X_evaluation, y_calibration, y_evaluation = (
        train_test_split(
            X,
            y,
            test_size=0.50,
            random_state=42,
            stratify=y
        )
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
    # 4. Generate XGBoost probabilities
    # ========================================================

    print(
        "\nGenerating XGBoost probabilities..."
    )

    calibration_probabilities = (
        model.predict_proba(
            X_calibration
        )[:, 1]
    )

    evaluation_probabilities = (
        model.predict_proba(
            X_evaluation
        )[:, 1]
    )

    # ========================================================
    # 5. Fit Platt scaling
    # ========================================================

    print(
        "\nTraining Platt calibration..."
    )

    calibration_scores = np.log(
        np.clip(
            calibration_probabilities,
            1e-6,
            1 - 1e-6
        )
        /
        (
            1
            -
            np.clip(
                calibration_probabilities,
                1e-6,
                1 - 1e-6
            )
        )
    ).reshape(-1, 1)

    platt_calibrator = LogisticRegression(
        random_state=42
    )

    platt_calibrator.fit(
        calibration_scores,
        y_calibration
    )

    print(
        "Platt calibration completed."
    )

    # ========================================================
    # 6. Apply Platt scaling
    # ========================================================

    evaluation_scores = np.log(
        np.clip(
            evaluation_probabilities,
            1e-6,
            1 - 1e-6
        )
        /
        (
            1
            -
            np.clip(
                evaluation_probabilities,
                1e-6,
                1 - 1e-6
            )
        )
    ).reshape(-1, 1)

    calibrated_probabilities = (
        platt_calibrator.predict_proba(
            evaluation_scores
        )[:, 1]
    )

    # ========================================================
    # 7. Evaluate BEFORE calibration
    # ========================================================

    before = {
        "roc_auc": roc_auc_score(
            y_evaluation,
            evaluation_probabilities
        ),
        "pr_auc": average_precision_score(
            y_evaluation,
            evaluation_probabilities
        ),
        "ks": calculate_ks(
            y_evaluation,
            evaluation_probabilities
        ),
        "calibration_error": calculate_calibration_error(
            y_evaluation,
            evaluation_probabilities
        ),
    }

    # ========================================================
    # 8. Evaluate AFTER calibration
    # ========================================================

    after = {
        "roc_auc": roc_auc_score(
            y_evaluation,
            calibrated_probabilities
        ),
        "pr_auc": average_precision_score(
            y_evaluation,
            calibrated_probabilities
        ),
        "ks": calculate_ks(
            y_evaluation,
            calibrated_probabilities
        ),
        "calibration_error": calculate_calibration_error(
            y_evaluation,
            calibrated_probabilities
        ),
    }

    # ========================================================
    # 9. Display comparison
    # ========================================================

    print("\n" + "=" * 60)
    print("CALIBRATION COMPARISON")
    print("=" * 60)

    print(
        f"{'Metric':<22}"
        f"{'Before':>12}"
        f"{'After':>12}"
    )

    print("-" * 46)

    for metric in before:

        print(
            f"{metric:<22}"
            f"{before[metric]:>12.4f}"
            f"{after[metric]:>12.4f}"
        )

    # ========================================================
    # 10. Calibration curves
    # ========================================================

    before_fraction, before_mean = calibration_curve(
        y_evaluation,
        evaluation_probabilities,
        n_bins=10,
        strategy="quantile"
    )

    after_fraction, after_mean = calibration_curve(
        y_evaluation,
        calibrated_probabilities,
        n_bins=10,
        strategy="quantile"
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        before_mean,
        before_fraction,
        marker="o",
        label="Before Platt scaling"
    )

    plt.plot(
        after_mean,
        after_fraction,
        marker="o",
        label="After Platt scaling"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration"
    )

    plt.xlabel(
        "Mean predicted probability"
    )

    plt.ylabel(
        "Observed default rate"
    )

    plt.title(
        "CreditIQ Calibration Comparison"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_FILE,
        dpi=150
    )

    plt.close()

    # ========================================================
    # 11. Save calibrator
    # ========================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        platt_calibrator,
        CALIBRATOR_FILE
    )

    # ========================================================
    # 12. Save results
    # ========================================================

    results = {
        "before_calibration": before,
        "after_platt_scaling": after,
        "calibration_method": "Platt scaling",
        "calibration_improved": (
            after["calibration_error"]
            < before["calibration_error"]
        ),
    }

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

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
    # 13. Final output
    # ========================================================

    print(
        f"\nSaved Platt calibrator to:\n"
        f"{CALIBRATOR_FILE}"
    )

    print(
        f"\nSaved calibration results to:\n"
        f"{METRICS_FILE}"
    )

    print(
        f"\nSaved calibration comparison to:\n"
        f"{PLOT_FILE}"
    )

    print(
        "\nCreditIQ probability calibration "
        "completed successfully."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    run_calibration()