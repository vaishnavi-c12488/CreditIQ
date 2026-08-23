"""
CreditIQ — SHAP Explainability

Explains the finalized Monotonic XGBoost model.

Important:
- No model retraining.
- No threshold tuning.
- No test-set model selection.
- Uses validation data for explainability.
- Explains the model before the final decision threshold.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "xgboost_monotonic.joblib"
)

VALIDATION_FILE = (
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

FIGURES_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
)

METRICS_DIR = (
    PROJECT_ROOT
    / "reports"
    / "metrics"
)

IMPORTANCE_FILE = (
    METRICS_DIR
    / "shap_feature_importance.csv"
)

# Number of validation observations used for SHAP.
# This keeps explainability computationally manageable.
SHAP_SAMPLE_SIZE = 5000

RANDOM_STATE = 42

TARGET = "TARGET"
ID_COLUMN = "SK_ID_CURR"


# ============================================================
# MAIN
# ============================================================

def run_shap_analysis():

    print(
        "Starting CreditIQ SHAP explainability..."
    )

    # ========================================================
    # 1. CHECK FILES
    # ========================================================

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_FILE}"
        )

    if not VALIDATION_FILE.exists():

        raise FileNotFoundError(
            f"Validation dataset not found:\n"
            f"{VALIDATION_FILE}"
        )

    if not SELECTED_FEATURES_FILE.exists():

        raise FileNotFoundError(
            f"Selected feature file not found:\n"
            f"{SELECTED_FEATURES_FILE}"
        )

    # ========================================================
    # 2. LOAD MODEL
    # ========================================================

    print(
        "\nLoading final Monotonic XGBoost..."
    )

    model = joblib.load(
        MODEL_FILE
    )

    # ========================================================
    # 3. LOAD VALIDATION DATA
    # ========================================================

    print(
        "Reading validation dataset..."
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
    # 4. LOAD SELECTED FEATURES
    # ========================================================

    selected_features_df = pd.read_csv(
        SELECTED_FEATURES_FILE
    )

    selected_features = (
        selected_features_df[
            "feature"
        ]
        .tolist()
    )

    print(
        f"Selected features: "
        f"{len(selected_features)}"
    )

    # ========================================================
    # 5. VERIFY FEATURES
    # ========================================================

    missing_features = [
        feature
        for feature in selected_features
        if feature not in validation.columns
    ]

    if missing_features:

        raise ValueError(
            "Selected features missing from "
            "validation dataset:\n"
            f"{missing_features}"
        )

    # ========================================================
    # 6. PREPARE VALIDATION FEATURES
    # ========================================================

    X_validation = validation[
        selected_features
    ].copy()

    # ========================================================
    # 7. SAMPLE VALIDATION DATA
    # ========================================================

    if len(X_validation) > SHAP_SAMPLE_SIZE:

        X_shap = X_validation.sample(
            n=SHAP_SAMPLE_SIZE,
            random_state=RANDOM_STATE
        )

    else:

        X_shap = X_validation.copy()

    print(
        f"SHAP observations: "
        f"{len(X_shap):,}"
    )

    # ========================================================
    # 8. CREATE SHAP EXPLAINER
    # ========================================================

    print(
        "\nCreating SHAP TreeExplainer..."
    )

    explainer = shap.TreeExplainer(
        model
    )

    # ========================================================
    # 9. CALCULATE SHAP VALUES
    # ========================================================

    print(
        "Calculating SHAP values..."
    )

    shap_values = explainer.shap_values(
        X_shap
    )

    # ========================================================
    # 10. HANDLE SHAP OUTPUT FORMAT
    # ========================================================

    # Some SHAP/model combinations can return
    # an Explanation object or a list.
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

    # ========================================================
    # 11. VALIDATE SHAPE
    # ========================================================

    if shap_values.ndim != 2:

        raise ValueError(
            "Unexpected SHAP value shape: "
            f"{shap_values.shape}"
        )

    if shap_values.shape[1] != len(
        selected_features
    ):

        raise ValueError(
            "SHAP feature count does not match "
            "selected feature count."
        )

    # ========================================================
    # 12. GLOBAL SHAP IMPORTANCE
    # ========================================================

    mean_abs_shap = np.mean(
        np.abs(
            shap_values
        ),
        axis=0
    )

    importance = pd.DataFrame(
        {
            "feature": selected_features,
            "mean_abs_shap": mean_abs_shap
        }
    )

    importance = (
        importance
        .sort_values(
            "mean_abs_shap",
            ascending=False
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # 13. SAVE IMPORTANCE TABLE
    # ========================================================

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    importance.to_csv(
        IMPORTANCE_FILE,
        index=False
    )

    print(
        "\nSaved SHAP feature importance to:"
    )

    print(
        IMPORTANCE_FILE
    )

    # ========================================================
    # 14. DISPLAY TOP FEATURES
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "TOP 20 FEATURES BY MEAN ABSOLUTE SHAP"
    )

    print(
        "=" * 60
    )

    print(
        importance.head(20).to_string(
            index=False
        )
    )

    # ========================================================
    # 15. CREATE FIGURES DIRECTORY
    # ========================================================

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # 16. SHAP BAR SUMMARY
    # ========================================================

    print(
        "\nCreating SHAP bar summary..."
    )

    plt.figure(
        figsize=(10, 8)
    )

    shap.summary_plot(
        shap_values,
        X_shap,
        plot_type="bar",
        max_display=20,
        show=False
    )

    plt.tight_layout()

    bar_file = (
        FIGURES_DIR
        / "shap_summary_bar.png"
    )

    plt.savefig(
        bar_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved:\n{bar_file}"
    )

    # ========================================================
    # 17. SHAP BEESWARM
    # ========================================================

    print(
        "\nCreating SHAP beeswarm..."
    )

    plt.figure(
        figsize=(10, 8)
    )

    shap.summary_plot(
        shap_values,
        X_shap,
        max_display=20,
        show=False
    )

    plt.tight_layout()

    beeswarm_file = (
        FIGURES_DIR
        / "shap_summary_beeswarm.png"
    )

    plt.savefig(
        beeswarm_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved:\n{beeswarm_file}"
    )

    # ========================================================
    # 18. DEPENDENCE PLOTS FOR TOP FEATURES
    # ========================================================

    top_features = (
        importance
        .head(5)
        ["feature"]
        .tolist()
    )

    print(
        "\nCreating SHAP dependence plots..."
    )

    for feature in top_features:

        try:

            plt.figure(
                figsize=(9, 6)
            )

            shap.dependence_plot(
                feature,
                shap_values,
                X_shap,
                interaction_index=None,
                show=False
            )

            plt.tight_layout()

            safe_name = (
                feature
                .replace("/", "_")
                .replace("\\", "_")
                .replace(" ", "_")
            )

            output_file = (
                FIGURES_DIR
                / f"shap_dependence_{safe_name}.png"
            )

            plt.savefig(
                output_file,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

            print(
                f"Saved: {output_file}"
            )

        except Exception as error:

            print(
                f"WARNING: Could not create "
                f"dependence plot for "
                f"{feature}: {error}"
            )

    # ========================================================
    # 19. SAVE SUMMARY JSON
    # ========================================================

    summary_file = (
        METRICS_DIR
        / "shap_summary.json"
    )

    summary = {

        "model":
            "Monotonic XGBoost",

        "dataset":
            "validation_final.parquet",

        "dataset_role":
            "validation explainability only",

        "shap_sample_size":
            int(len(X_shap)),

        "selected_features":
            int(len(selected_features)),

        "top_features":
            importance
            .head(20)
            .to_dict(
                orient="records"
            ),

        "test_set_used":
            False,

        "model_retrained":
            False,

        "threshold_tuned":
            False
    }

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as file:

        import json

        json.dump(
            summary,
            file,
            indent=4
        )

    print(
        "\nSaved SHAP summary to:"
    )

    print(
        summary_file
    )

    # ========================================================
    # 20. COMPLETION
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "CreditIQ SHAP explainability completed successfully."
    )

    print(
        "=" * 60
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_shap_analysis()