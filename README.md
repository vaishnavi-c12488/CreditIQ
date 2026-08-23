# CreditIQ

## Credit Risk Scoring & Decision Support System

CreditIQ is an end-to-end machine learning project for credit-risk assessment, built to go beyond simply training a classification model.

The system takes applicant and credit-history data, creates applicant-level features from multiple relational sources, predicts the probability of default using a monotonic XGBoost model, calibrates that probability, converts it into a decision using a frozen threshold, explains the prediction with SHAP, evaluates fairness, and serves the complete workflow through FastAPI, PostgreSQL, Streamlit, MLflow, and Docker.

The main idea behind CreditIQ is simple:

> **A useful ML system should not only make predictions. It should also explain them, evaluate their reliability, consider fairness, and be deployable.**

---
## Final Model at a Glance

| Component | Final Choice |
|---|---|
| **Model** | Monotonic XGBoost |
| **Base Algorithm** | XGBoost Gradient Boosted Trees |
| **Model Features** | 80 |
| **Training Dataset** | 307,511 applicants |
| **Calibration** | Platt Scaling |
| **Decision Threshold** | 0.20 |
| **Explainability** | SHAP TreeExplainer |
| **Fairness Evaluation** | Fairlearn |
| **Model Tracking** | MLflow |
| **Deployment** | FastAPI + Docker |
| **Database** | PostgreSQL |

### Final Test Performance

| Metric | Score |
|---|---:|
| **ROC-AUC** | **0.7630** |
| **PR-AUC** | **0.2826** |
| **KS Statistic** | **0.3942** |
| **Precision** | **31.38%** |
| **Recall** | **34.17%** |
| **F1 Score** | **32.71%** |
| **Calibration Error** | **0.0096** |
| **False Positive Rate** | **8.44%** |

> The final model is a monotonic XGBoost classifier selected after comparison with Logistic Regression and unconstrained XGBoost. The probability outputs are calibrated using Platt scaling, and a validation-selected threshold of 0.20 is used for the final approval/decline decision.

## Project Highlights

| | |
|---|---|
| **307,511** | Applicants processed |
| **80** | Model features |
| **0.7630** | Final ROC-AUC |
| **0.2826** | Final PR-AUC |
| **0.0096** | Final calibration error |
| **0.20** | Frozen decision threshold |
| **5,000** | Validation observations used for SHAP analysis |

---

## Why CreditIQ?

Credit-risk modeling is more complicated than predicting `0` or `1`.

A model may have good predictive performance but still have problems:

- Its probabilities may not be reliable.
- Its decisions may be difficult to explain.
- Its behavior may differ across applicant groups.
- A small improvement in AUC may come at the cost of less defensible model behavior.
- A model sitting inside a notebook is not the same as a deployable ML system.

CreditIQ was built around these problems.

Instead of treating the project as:

```text
Dataset → Model → Prediction

Data
  ↓
ETL & Feature Engineering
  ↓
WOE / IV
  ↓
Model Comparison
  ↓
Monotonic XGBoost
  ↓
Probability Calibration
  ↓
Decision Threshold
  ↓
Explainability
  ↓
Fairness Evaluation
  ↓
API Serving
  ↓
Database + Audit Logs
  ↓
Dashboard
  ↓
Docker Deployment

                         ┌──────────────────────┐
                         │   Home Credit Data   │
                         │   7 Related Tables   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Data Ingestion    │
                         │      + ETL           │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     PostgreSQL       │
                         │   Raw + Feature DB   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Feature Engineering  │
                         │       WOE / IV       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                  ┌──────────────────────────────────┐
                  │        MODEL DEVELOPMENT         │
                  │                                  │
                  │ Logistic Regression              │
                  │ XGBoost                          │
                  │ Monotonic XGBoost                │
                  └────────────────┬─────────────────┘
                                   │
                                   ▼
                         ┌──────────────────────┐
                         │  Platt Calibration   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Frozen Threshold     │
                         │        0.20          │
                         └──────────┬───────────┘
                                    │
                  ┌─────────────────┼─────────────────┐
                  │                 │                 │
                  ▼                 ▼                 ▼
          ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
          │     SHAP     │  │  Fairlearn   │  │   MLflow     │
          │ Explainability│ │   Fairness   │  │   Tracking   │
          └───────┬──────┘  └──────────────┘  └──────────────┘
                  │
                  ▼
          ┌──────────────────┐
          │      FastAPI     │
          │ /score /explain  │
          │     /health      │
          └────────┬─────────┘
                   │
             ┌─────┴─────┐
             ▼           ▼
      ┌────────────┐ ┌────────────┐
      │ PostgreSQL │ │ Streamlit  │
      │ Audit Logs │ │ Dashboard  │
      └────────────┘ └────────────┘

                  Docker Compose