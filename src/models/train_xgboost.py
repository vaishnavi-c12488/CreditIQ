"""
CreditIQ — Unconstrained XGBoost

This script:
1. Loads the final WOE-encoded train and validation datasets.
2. Separates features (X) from the target (y).
3. Trains an unconstrained XGBoost classifier.
4. Evaluates the model on the validation set.
5. Saves the trained model and validation metrics.

The test dataset is intentionally not used at this stage.
"""

from pathlib import Path
import json

import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
import joblib


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"

TRAIN_FILE = PROCESSED_DIR / "train_final.parquet"
VALIDATION_FILE = PROCESSED_DIR / "validation_final.parquet"

MODEL_FILE = MODEL_DIR / "xgboost_unconstrained.joblib"
METRICS_FILE = METRICS_DIR / "xgboost_unconstrained.json"


# ============================================================
# Training
# ============================================================

def run_xgboost():

    print("Starting CreditIQ XGBoost model...")

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("Reading final WOE datasets...")

    train = pd.read_parquet(TRAIN_FILE)
    validation = pd.read_parquet(VALIDATION_FILE)

    print(
        f"Train: {len(train):,} rows × "
        f"{len(train.columns):,} columns"
    )

    print(
        f"Validation: {len(validation):,} rows × "
        f"{len(validation.columns):,} columns"
    )

    # --------------------------------------------------------
    # Separate X and y
    # --------------------------------------------------------

    TARGET = "TARGET"
    ID_COLUMN = "SK_ID_CURR"

    feature_columns = [
        col
        for col in train.columns
        if col not in [TARGET, ID_COLUMN]
    ]

    X_train = train[feature_columns]
    y_train = train[TARGET]

    X_validation = validation[feature_columns]
    y_validation = validation[TARGET]

    print(
        f"Training features: {len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Class imbalance
    # --------------------------------------------------------

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    scale_pos_weight = (
        negative_count / positive_count
    )

    print(
        f"Negative class: {negative_count:,}"
    )

    print(
        f"Positive class: {positive_count:,}"
    )

    print(
        f"scale_pos_weight: {scale_pos_weight:.4f}"
    )

    # --------------------------------------------------------
    # XGBoost model
    # --------------------------------------------------------

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nTraining XGBoost...")

    model.fit(
        X_train,
        y_train
    )

    print("Model training completed.")

    # --------------------------------------------------------
    # Validation predictions
    # --------------------------------------------------------

    print("\nEvaluating on validation data...")

    y_pred = model.predict(
        X_validation
    )

    y_probability = model.predict_proba(
        X_validation
    )[:, 1]

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("XGBOOST VALIDATION RESULTS")
    print("=" * 60)

    for metric, value in metrics.items():

        print(
            f"{metric.upper():10s}: "
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

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

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

    print(
        f"\nSaved model to:\n{MODEL_FILE}"
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    print(
        f"\nSaved validation metrics to:\n"
        f"{METRICS_FILE}"
    )

    print(
        "\nCreditIQ XGBoost model completed successfully."
    )


if __name__ == "__main__":
    run_xgboost()