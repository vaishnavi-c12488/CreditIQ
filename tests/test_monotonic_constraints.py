"""
CreditIQ — Monotonic Constraint Verification

Checks that increasing each constrained WOE feature
does not increase predicted default risk.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_FILE = PROJECT_ROOT / "models" / "xgboost_monotonic.joblib"


# ============================================================
# Configuration
# ============================================================

CONSTRAINED_FEATURES = {
    "EXT_SOURCE_1": -1,
    "EXT_SOURCE_2": -1,
    "EXT_SOURCE_3": -1,
}


# ============================================================
# Load model and validation data
# ============================================================

print("Loading monotonic XGBoost model...")

model = joblib.load(MODEL_FILE)

validation = pd.read_parquet(
    PROCESSED_DIR / "validation_final.parquet"
)

feature_columns = [
    column
    for column in validation.columns
    if column not in ["TARGET", "SK_ID_CURR"]
]

X_validation = validation[feature_columns].copy()

print(
    f"Validation rows: {len(X_validation):,}"
)

print(
    f"Features: {len(feature_columns)}"
)


# ============================================================
# Select one representative applicant
# ============================================================

base_row = X_validation.iloc[[0]].copy()


# ============================================================
# Monotonicity sweep
# ============================================================

print("\n" + "=" * 60)
print("MONOTONICITY SWEEP TEST")
print("=" * 60)


all_passed = True


for feature, direction in CONSTRAINED_FEATURES.items():

    print(f"\nTesting: {feature}")

    original_value = base_row[feature].iloc[0]

    # Use validation distribution to create realistic values
    values = np.linspace(
        X_validation[feature].quantile(0.05),
        X_validation[feature].quantile(0.95),
        10
    )

    predictions = []

    for value in values:

        test_row = base_row.copy()

        test_row[feature] = value

        probability = model.predict_proba(
            test_row
        )[0, 1]

        predictions.append(probability)

    predictions = np.array(predictions)

    # --------------------------------------------------------
    # Check direction
    # --------------------------------------------------------

    differences = np.diff(predictions)

    if direction == -1:

        violations = (
            differences > 1e-8
        )

    elif direction == 1:

        violations = (
            differences < -1e-8
        )

    else:

        violations = np.zeros(
            len(differences),
            dtype=bool
        )

    violation_count = violations.sum()

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print(
        f"Original value: {original_value:.6f}"
    )

    print("\nSweep values and predicted risk:")

    for value, probability in zip(
        values,
        predictions
    ):

        print(
            f"  {value:.6f} → "
            f"{probability:.6f}"
        )

    print(
        f"\nViolations: {violation_count}"
    )

    if violation_count == 0:

        print(
            f"PASS: {feature} "
            f"respects its monotonic constraint."
        )

    else:

        print(
            f"FAIL: {feature} "
            f"has monotonicity violations."
        )

        all_passed = False


# ============================================================
# Final result
# ============================================================

print("\n" + "=" * 60)
print("MONOTONICITY TEST SUMMARY")
print("=" * 60)

if all_passed:

    print(
        "PASS — All monotonic constraints "
        "were satisfied."
    )

else:

    print(
        "FAIL — One or more monotonic "
        "constraints were violated."
    )