"""
CreditIQ — MLflow Tracking

Logs completed CreditIQ model-development, calibration,
fairness, and final-evaluation artifacts to MLflow.

This script does NOT retrain any model.
"""

from pathlib import Path
import json
import os

import mlflow


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
CONFIG_DIR = PROJECT_ROOT / "configs"


# ============================================================
# MLflow configuration
# ============================================================

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000"
)
EXPERIMENT_NAME = "CreditIQ"


# ============================================================
# Utility functions
# ============================================================

def load_json(path: Path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def log_numeric_metrics(
    metrics: dict,
    prefix: str = ""
):
    for key, value in metrics.items():

        if isinstance(value, (int, float)):

            mlflow.log_metric(
                f"{prefix}{key}",
                float(value)
            )


def log_json_artifact(path: Path):

    if path.exists():
        mlflow.log_artifact(
            str(path),
            artifact_path="metrics"
        )


def log_model_artifact(path: Path):

    if path.exists():
        mlflow.log_artifact(
            str(path),
            artifact_path="models"
        )


def log_figure(path: Path):

    if path.exists():
        mlflow.log_artifact(
            str(path),
            artifact_path="figures"
        )


# ============================================================
# MLflow setup
# ============================================================

def setup_mlflow():

    mlflow.set_tracking_uri(
        TRACKING_URI
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    print(
        f"MLflow tracking URI: "
        f"{mlflow.get_tracking_uri()}"
    )

    print(
        f"MLflow experiment: "
        f"{EXPERIMENT_NAME}"
    )


# ============================================================
# 1. Logistic Regression baseline
# ============================================================

def log_baseline_run():

    metrics = load_json(
        METRICS_DIR
        / "baseline_logistic_regression.json"
    )

    with mlflow.start_run(
        run_name="baseline_logistic_regression"
    ):

        mlflow.set_tags(
            {
                "model_type": "Logistic Regression",
                "stage": "model_comparison",
                "dataset": "validation",
                "test_set_used": "false",
            }
        )

        mlflow.log_params(
            {
                "model": "Logistic Regression",
                "class_weight": "balanced",
                "max_iter": 1000,
                "random_state": 42,
            }
        )

        log_numeric_metrics(
            metrics
        )

        log_json_artifact(
            METRICS_DIR
            / "baseline_logistic_regression.json"
        )

        log_model_artifact(
            MODEL_DIR
            / "baseline_logistic_regression.joblib"
        )

    print(
        "Logged baseline Logistic Regression run."
    )


# ============================================================
# 2. Unconstrained XGBoost
# ============================================================

def log_unconstrained_run():

    metrics = load_json(
        METRICS_DIR
        / "xgboost_unconstrained.json"
    )

    with mlflow.start_run(
        run_name="xgboost_unconstrained"
    ):

        mlflow.set_tags(
            {
                "model_type": "XGBoost",
                "constraints": "none",
                "stage": "model_comparison",
                "dataset": "validation",
                "test_set_used": "false",
            }
        )

        mlflow.log_params(
            {
                "model": "XGBoost",
                "n_estimators": 300,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "random_state": 42,
            }
        )

        log_numeric_metrics(
            metrics
        )

        log_json_artifact(
            METRICS_DIR
            / "xgboost_unconstrained.json"
        )

        log_model_artifact(
            MODEL_DIR
            / "xgboost_unconstrained.joblib"
        )

    print(
        "Logged unconstrained XGBoost run."
    )


# ============================================================
# 3. Monotonic XGBoost
# ============================================================

def log_monotonic_run():

    metrics = load_json(
        METRICS_DIR
        / "xgboost_monotonic.json"
    )

    with mlflow.start_run(
        run_name="xgboost_monotonic"
    ):

        mlflow.set_tags(
            {
                "model_type": "XGBoost",
                "constraints": "monotonic",
                "stage": "final_model_selection",
                "dataset": "validation",
                "test_set_used": "false",
            }
        )

        mlflow.log_params(
            {
                "model": "Monotonic XGBoost",
                "n_estimators": 300,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "monotonic_features": (
                    "EXT_SOURCE_1,"
                    "EXT_SOURCE_2,"
                    "EXT_SOURCE_3"
                ),
                "random_state": 42,
            }
        )

        log_numeric_metrics(
            metrics
        )

        log_json_artifact(
            METRICS_DIR
            / "xgboost_monotonic.json"
        )

        log_model_artifact(
            MODEL_DIR
            / "xgboost_monotonic.joblib"
        )

    print(
        "Logged monotonic XGBoost run."
    )


# ============================================================
# 4. Calibration
# ============================================================

def log_calibration_run():

    results = load_json(
        METRICS_DIR
        / "calibration_results.json"
    )

    before = results[
        "before_calibration"
    ]

    after = results[
        "after_platt_scaling"
    ]

    with mlflow.start_run(
        run_name="platt_calibration"
    ):

        mlflow.set_tags(
            {
                "stage": "calibration",
                "method": "Platt scaling",
                "dataset": "validation",
                "test_set_used": "false",
            }
        )

        mlflow.log_params(
            {
                "calibration_method": (
                    "Platt scaling"
                ),
                "base_model": (
                    "Monotonic XGBoost"
                ),
            }
        )

        log_numeric_metrics(
            before,
            prefix="before_"
        )

        log_numeric_metrics(
            after,
            prefix="after_"
        )

        mlflow.log_metric(
            "calibration_error_improvement",
            float(
                before["calibration_error"]
                - after["calibration_error"]
            )
        )

        log_json_artifact(
            METRICS_DIR
            / "calibration_results.json"
        )

        log_model_artifact(
            MODEL_DIR
            / "platt_calibrator.joblib"
        )

        log_figure(
            FIGURES_DIR
            / "calibration_curve_before.png"
        )

        log_figure(
            FIGURES_DIR
            / "calibration_curve.png"
        )

        log_figure(
            FIGURES_DIR
            / "calibration_curve_comparison.png"
        )

    print(
        "Logged calibration run."
    )


# ============================================================
# 5. Fairness audit
# ============================================================

def log_fairness_run():

    results = load_json(
        METRICS_DIR
        / "fairness_audit.json"
    )

    before = results[
        "before_mitigation"
    ]

    after = results[
        "after_mitigation"
    ]

    with mlflow.start_run(
        run_name="fairness_audit"
    ):

        mlflow.set_tags(
            {
                "stage": "fairness_audit",
                "model": "Monotonic XGBoost",
                "dataset": "validation",
                "test_set_used": "false",
                "primary_metric": (
                    "equalized_odds_difference"
                ),
            }
        )

        mlflow.log_params(
            {
                "age_proxy": "DAYS_BIRTH",
                "groups": (
                    "<25,25-34,35-44,45-54,55+"
                ),
                "threshold": 0.20,
                "mitigation_method": (
                    "Fairlearn ThresholdOptimizer"
                ),
                "constraint": "equalized_odds",
                "objective": "balanced_accuracy_score",
            }
        )

        mlflow.log_metric(
            "before_demographic_parity_difference",
            float(
                before["fairness"][
                    "demographic_parity_difference"
                ]
            )
        )

        mlflow.log_metric(
            "before_equalized_odds_difference",
            float(
                before["fairness"][
                    "equalized_odds_difference"
                ]
            )
        )

        mlflow.log_metric(
            "after_demographic_parity_difference",
            float(
                after["fairness"][
                    "demographic_parity_difference"
                ]
            )
        )

        mlflow.log_metric(
            "after_equalized_odds_difference",
            float(
                after["fairness"][
                    "equalized_odds_difference"
                ]
            )
        )

        log_json_artifact(
            METRICS_DIR
            / "fairness_audit.json"
        )

    print(
        "Logged fairness audit run."
    )


# ============================================================
# 6. Final test evaluation
# ============================================================

def log_final_test_run():

    results = load_json(
        METRICS_DIR
        / "final_test_results.json"
    )

    metrics = results[
        "metrics"
    ]

    with mlflow.start_run(
        run_name="final_test_evaluation"
    ):

        mlflow.set_tags(
            {
                "stage": "final_test_evaluation",
                "model": "Monotonic XGBoost",
                "calibration": "Platt scaling",
                "dataset": "test",
                "test_set_used": "final_evaluation_only",
                "threshold_frozen": "true",
            }
        )

        mlflow.log_params(
            {
                "model": "Monotonic XGBoost",
                "calibration": "Platt scaling",
                "threshold": 0.20,
                "threshold_source": (
                    "configs/threshold.yaml"
                ),
                "threshold_selection_data": (
                    "validation"
                ),
                "test_rows": 46127,
                "test_features": 80,
            }
        )

        log_numeric_metrics(
            metrics
        )

        log_json_artifact(
            METRICS_DIR
            / "final_test_results.json"
        )

        log_json_artifact(
            CONFIG_DIR
            / "threshold.yaml"
        )

        log_model_artifact(
            MODEL_DIR
            / "xgboost_monotonic.joblib"
        )

    print(
        "Logged final test evaluation run."
    )


# ============================================================
# 7. SHAP artifacts
# ============================================================

def log_shap_artifacts():

    with mlflow.start_run(
        run_name="shap_explainability"
    ):

        mlflow.set_tags(
            {
                "stage": "explainability",
                "model": "Monotonic XGBoost",
                "dataset": "validation",
                "test_set_used": "false",
            }
        )

        mlflow.log_params(
            {
                "explainer": "TreeExplainer",
                "observations": 5000,
                "selected_features": 80,
            }
        )

        for filename in [
            "shap_summary.json",
            "shap_feature_importance.csv",
            "applicant_shap_examples.json",
        ]:

            log_json_artifact(
                METRICS_DIR / filename
            )

        for filename in [
            "shap_summary_bar.png",
            "shap_summary_beeswarm.png",
            "shap_dependence_EXT_SOURCE_1.png",
            "shap_dependence_EXT_SOURCE_2.png",
            "shap_dependence_EXT_SOURCE_3.png",
            "shap_dependence_installment_ontime_ratio.png",
            "shap_dependence_previous_avg_credit_to_application_ratio.png",
        ]:

            log_figure(
                FIGURES_DIR / filename
            )

    print(
        "Logged SHAP artifacts."
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("CreditIQ — MLflow Tracking")
    print("=" * 60)

    setup_mlflow()

    log_baseline_run()
    log_unconstrained_run()
    log_monotonic_run()
    log_calibration_run()
    log_fairness_run()
    log_final_test_run()
    log_shap_artifacts()

    print("\n" + "=" * 60)
    print(
        "CreditIQ MLflow tracking completed successfully."
    )
    print("=" * 60)


if __name__ == "__main__":
    main()