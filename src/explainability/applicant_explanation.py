"""
CreditIQ — Per-Applicant SHAP Explanation

Generates applicant-level SHAP reason codes for the finalized
Monotonic XGBoost model.

The explanation uses validation/feature-store data only.
No model retraining or threshold tuning is performed.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import shap
import yaml


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "xgboost_monotonic.joblib"
)

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation_final.parquet"
)

SELECTED_FEATURES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "selected_features.csv"
)

THRESHOLD_FILE = (
    PROJECT_ROOT
    / "configs"
    / "threshold.yaml"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "metrics"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "applicant_shap_examples.json"
)

ID_COLUMN = "SK_ID_CURR"
TARGET = "TARGET"


# ============================================================
# Load configuration
# ============================================================

def load_threshold():

    with open(
        THRESHOLD_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        config = yaml.safe_load(file)

    return float(
        config["decision_threshold"]
    )


# ============================================================
# Load selected features
# ============================================================

def load_selected_features():

    selected = pd.read_csv(
        SELECTED_FEATURES_FILE
    )

    return selected["feature"].tolist()


# ============================================================
# Explain one applicant
# ============================================================

def explain_applicant(
    applicant_id,
    model,
    explainer,
    feature_data,
    selected_features,
    threshold,
    top_n=5
):

    applicant = feature_data[
        feature_data[ID_COLUMN] == applicant_id
    ].copy()

    if applicant.empty:

        raise KeyError(
            f"Applicant {applicant_id} was not found."
        )

    X = applicant[
        selected_features
    ]

    # --------------------------------------------------------
    # Model probability
    # --------------------------------------------------------

    probability = float(
        model.predict_proba(X)[0, 1]
    )

    decision = (
        "DECLINE"
        if probability >= threshold
        else "APPROVE"
    )

    # --------------------------------------------------------
    # SHAP explanation
    # --------------------------------------------------------

    shap_values = explainer.shap_values(X)

    if isinstance(
        shap_values,
        list
    ):

        shap_values = shap_values[1]

    if hasattr(
        shap_values,
        "values"
    ):

        shap_values = shap_values.values

    shap_values = np.asarray(
        shap_values
    )

    shap_row = shap_values[0]

    # --------------------------------------------------------
    # Build contribution table
    # --------------------------------------------------------

    contributions = pd.DataFrame(
        {
            "feature": selected_features,
            "value": X.iloc[0].values,
            "shap_value": shap_row,
        }
    )

    contributions[
        "absolute_shap"
    ] = np.abs(
        contributions["shap_value"]
    )

    # Positive SHAP → increases model output/risk.
    positive = (
        contributions[
            contributions["shap_value"] > 0
        ]
        .sort_values(
            "shap_value",
            ascending=False
        )
        .head(top_n)
    )

    # Negative SHAP → decreases model output/risk.
    negative = (
        contributions[
            contributions["shap_value"] < 0
        ]
        .sort_values(
            "shap_value",
            ascending=True
        )
        .head(top_n)
    )

    # --------------------------------------------------------
    # Convert to JSON-safe records
    # --------------------------------------------------------

    def records_to_json(frame):

        records = []

        for _, row in frame.iterrows():

            value = row["value"]

            if pd.isna(value):

                value = None

            elif isinstance(
                value,
                (np.integer,)
            ):

                value = int(value)

            elif isinstance(
                value,
                (np.floating,)
            ):

                value = float(value)

            records.append(
                {
                    "feature": row["feature"],
                    "value": value,
                    "shap_value": float(
                        row["shap_value"]
                    ),
                    "direction": (
                        "increases_risk"
                        if row["shap_value"] > 0
                        else "decreases_risk"
                    ),
                }
            )

        return records

    return {
        "applicant_id": int(applicant_id),
        "predicted_probability": probability,
        "decision_threshold": threshold,
        "decision": decision,
        "top_risk_increasing_factors":
            records_to_json(positive),
        "top_risk_reducing_factors":
            records_to_json(negative),
    }


# ============================================================
# Main validation
# ============================================================

def main():

    print("=" * 60)
    print("CreditIQ PER-APPLICANT SHAP EXPLANATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading Monotonic XGBoost...")

    model = joblib.load(
        MODEL_FILE
    )

    # --------------------------------------------------------
    # Load feature data
    # --------------------------------------------------------

    print(
        "Loading engineered feature store..."
    )

    feature_data = pd.read_parquet(
        FEATURE_FILE
    )

    # --------------------------------------------------------
    # Load selected features
    # --------------------------------------------------------

    selected_features = (
        load_selected_features()
    )

    print(
        f"Selected features: "
        f"{len(selected_features)}"
    )

    # --------------------------------------------------------
    # Validate selected features
    # --------------------------------------------------------

    missing = [
        feature
        for feature in selected_features
        if feature not in feature_data.columns
    ]

    if missing:

        raise ValueError(
            "Selected features missing from "
            f"feature store: {missing}"
        )

    # --------------------------------------------------------
    # Load frozen threshold
    # --------------------------------------------------------

    threshold = load_threshold()

    print(
        f"Frozen threshold: "
        f"{threshold:.4f}"
    )

    # --------------------------------------------------------
    # Create SHAP explainer
    # --------------------------------------------------------

    print(
        "\nCreating SHAP TreeExplainer..."
    )

    explainer = shap.TreeExplainer(
        model
    )

    # --------------------------------------------------------
    # Select three applicants
    # --------------------------------------------------------

    applicant_ids = (
        feature_data[
            ID_COLUMN
        ]
        .head(3)
        .tolist()
    )

    print(
        "\nGenerating explanations for "
        f"{len(applicant_ids)} applicants..."
    )

    explanations = []

    for applicant_id in applicant_ids:

        explanation = explain_applicant(
            applicant_id=applicant_id,
            model=model,
            explainer=explainer,
            feature_data=feature_data,
            selected_features=selected_features,
            threshold=threshold,
            top_n=5,
        )

        explanations.append(
            explanation
        )

        print("\n" + "-" * 60)

        print(
            f"Applicant: "
            f"{explanation['applicant_id']}"
        )

        print(
            f"Probability: "
            f"{explanation['predicted_probability']:.4f}"
        )

        print(
            f"Decision: "
            f"{explanation['decision']}"
        )

        print(
            "\nTop risk-increasing factors:"
        )

        for item in explanation[
            "top_risk_increasing_factors"
        ]:

            print(
                f"  {item['feature']}: "
                f"{item['shap_value']:+.6f}"
            )

        print(
            "\nTop risk-reducing factors:"
        )

        for item in explanation[
            "top_risk_reducing_factors"
        ]:

            print(
                f"  {item['feature']}: "
                f"{item['shap_value']:+.6f}"
            )

    # --------------------------------------------------------
    # Save examples
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output = {
        "model": "Monotonic XGBoost",
        "threshold": threshold,
        "feature_count": len(
            selected_features
        ),
        "test_set_used": False,
        "examples": explanations,
    }

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

    print("\n" + "=" * 60)

    print(
        "Saved applicant SHAP examples to:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nPer-applicant SHAP explanation "
        "completed successfully."
    )

    print("=" * 60)


if __name__ == "__main__":
    main()