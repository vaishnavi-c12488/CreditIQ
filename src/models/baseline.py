"""
CreditIQ - Baseline Logistic Regression

This script trains our first credit-risk model using the final
WOE-transformed features.

What we do:
- Load the final train and validation datasets.
- Separate features from TARGET.
- Handle missing values using the training data.
- Standardize the features.
- Train a Logistic Regression model.
- Evaluate it using credit-risk focused metrics.
- Save the model and validation metrics.

The test dataset is kept untouched for the final evaluation.
"""

from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train_final.parquet"
)

VAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation_final.parquet"
)

MODEL_DIR = PROJECT_ROOT / "models"

METRICS_DIR = (
    PROJECT_ROOT
    / "reports"
    / "metrics"
)

MODEL_FILE = (
    MODEL_DIR
    / "baseline_logistic_regression.joblib"
)

METRICS_FILE = (
    METRICS_DIR
    / "baseline_logistic_regression.json"
)


# ============================================================
# Main training pipeline
# ============================================================

def run_baseline():

    print("Starting CreditIQ baseline model...")

    # ========================================================
    # Load datasets
    # ========================================================

    print("Reading final WOE datasets...")

    train = pd.read_parquet(TRAIN_FILE)
    validation = pd.read_parquet(VAL_FILE)

    print(
        f"Train: {len(train):,} rows × "
        f"{len(train.columns):,} columns"
    )

    print(
        f"Validation: {len(validation):,} rows × "
        f"{len(validation.columns):,} columns"
    )

    # ========================================================
    # Separate features and target
    # ========================================================

    X_train = train.drop(
        columns=["SK_ID_CURR", "TARGET"]
    )

    y_train = train["TARGET"]

    X_validation = validation.drop(
        columns=["SK_ID_CURR", "TARGET"]
    )

    y_validation = validation["TARGET"]

    print(
        f"Training features: {X_train.shape[1]}"
    )

    # ========================================================
    # Logistic Regression pipeline
    # ========================================================

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42
                ),
            ),
        ]
    )

    # ========================================================
    # Train
    # ========================================================

    print("\nTraining Logistic Regression...")

    model.fit(
        X_train,
        y_train
    )

    print("Model training completed.")

    # ========================================================
    # Validation predictions
    # ========================================================

    print("\nEvaluating on validation data...")

    y_pred = model.predict(
        X_validation
    )

    y_probability = model.predict_proba(
        X_validation
    )[:, 1]

    # ========================================================
    # Metrics
    # ========================================================

    metrics = {
        "accuracy": accuracy_score(
            y_validation,
            y_pred
        ),
        "precision": precision_score(
            y_validation,
            y_pred,
            zero_division=0
        ),
        "recall": recall_score(
            y_validation,
            y_pred,
            zero_division=0
        ),
        "f1": f1_score(
            y_validation,
            y_pred,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_validation,
            y_probability
        ),
        "pr_auc": average_precision_score(
            y_validation,
            y_probability
        ),
    }

    # ========================================================
    # Print results
    # ========================================================

    print("\n" + "=" * 60)
    print("BASELINE LOGISTIC REGRESSION RESULTS")
    print("=" * 60)

    for metric, value in metrics.items():

        print(
            f"{metric.upper():<12}: "
            f"{value:.4f}"
        )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_validation,
            y_pred
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_validation,
            y_pred,
            zero_division=0
        )
    )

    # ========================================================
    # Save model
    # ========================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    # ========================================================
    # Save metrics
    # ========================================================

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    print("\nSaved model to:")
    print(MODEL_FILE)

    print("\nSaved validation metrics to:")
    print(METRICS_FILE)

    print(
        "\nCreditIQ baseline model "
        "completed successfully."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    run_baseline()