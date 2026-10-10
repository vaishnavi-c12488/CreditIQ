# CreditIQ

### Credit Risk Scoring & Decision Support System

CreditIQ is an end-to-end machine learning system designed for credit-risk assessment and responsible lending decisions. It goes beyond simply predicting whether an applicant may default by combining predictive modeling, explainability, fairness evaluation, audit logging, and production deployment.

The system processes applicant-level financial and credit information, performs feature engineering, and uses machine learning models to estimate the probability of default. The final model uses a monotonic XGBoost classifier with a frozen decision threshold selected during validation.

CreditIQ is designed as a production-oriented ML system with:

- Machine learning-based credit risk prediction
- Probability of default estimation
- Explainable AI using SHAP
- Applicant-level risk and protective factor analysis
- Fairness evaluation
- PostgreSQL-based feature storage and audit logging
- FastAPI-based model serving
- Streamlit-based risk assessment dashboard
- Docker-based deployment
- GitHub Actions CI pipeline
- MLflow experiment and model tracking
- Cloud deployment using Render

The system provides not only a credit decision, but also an explanation of **why** that decision was made, making the model more transparent, auditable, and suitable for responsible credit-risk applications.

---

## Final Model at a Glance

| Component | Final Choice |
|---|---|
| Model | Monotonic XGBoost |
| Base Algorithm | XGBoost Gradient Boosted Trees |
| Model Features | 80 |
| Training Dataset | 307,511 applicants |
| Database | PostgreSQL |
| Final Model Version | `xgboost_monotonic_v1` |
| Decision Threshold | 0.20 |
| Explainability | SHAP TreeExplainer |
| Fairness Evaluation | Fairness audit |
| API | FastAPI |
| Dashboard | Streamlit |
| Model Tracking | MLflow |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Cloud Deployment | Render |

---

## Final Test Performance

The final model was evaluated on a held-out test set using multiple classification, ranking, and calibration metrics.

| Metric | Score |
|---|---:|
| ROC-AUC | **0.7610** |
| PR-AUC | **0.2266** |
| Accuracy | **91.56%** |
| KS Statistic | **0.4202** |
| Precision | **31.20%** |
| Recall | **34.17%** |
| F1 Score | **32.71%** |
| Calibration Error | **0.0006** |
| False Positive Rate | **8.44%** |

### Why these metrics matter

Credit-risk prediction is an imbalanced classification problem, so accuracy alone is not enough to evaluate the model.

- **ROC-AUC** measures how well the model separates higher-risk and lower-risk applicants across thresholds.
- **PR-AUC** is particularly useful when the default class is relatively uncommon.
- **Precision** measures how many applicants predicted as risky were actually risky.
- **Recall** measures how many actual risky applicants were identified.
- **F1 Score** balances precision and recall.
- **KS Statistic** measures the separation between the distributions of good and bad applicants.
- **Calibration Error** evaluates how closely predicted probabilities reflect observed outcomes.
- **False Positive Rate** measures the proportion of non-defaulting applicants incorrectly classified as risky.

The final model was selected after comparing multiple modeling approaches and considering not only predictive performance, but also calibration, explainability, and monotonic behavior.

---

## Project Highlights

| Value | What It Represents |
|---|---|
| **307,511** | Applicants available in the production feature store |
| **80** | Model-ready applicant features |
| **0.7610** | Final ROC-AUC |
| **0.2266** | Final PR-AUC |
| **91.56%** | Final accuracy |
| **0.0006** | Calibration error |
| **0.20** | Frozen production decision threshold |
| **5,000** | Validation observations used for SHAP analysis |
| **XGBoost** | Final production model family |
| **SHAP** | Applicant-level explainability |
| **FastAPI** | Production model-serving API |
| **PostgreSQL** | Production feature store |
| **Render** | Cloud deployment platform |

---

## Architecture

CreditIQ follows an end-to-end production-oriented ML architecture that connects data engineering, feature engineering, model development, explainability, API serving, database storage, CI, and cloud deployment.

### System Architecture

```text
                         Home Credit Dataset
                                  │
                                  ▼
                        ┌──────────────────┐
                        │  Data Ingestion  │
                        │      + ETL       │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │ Feature          │
                        │ Engineering      │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │ Model Training   │
                        │ & Comparison     │
                        └────────┬─────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
             Logistic Regression       XGBoost Models
                                              │
                                              ▼
                                  Monotonic XGBoost
                                              │
                                              ▼
                                  Probability Calibration
                                       (Platt Scaling)
                                              │
                                              ▼
                                  Frozen Threshold = 0.20
                                              │
                           ┌──────────────────┴──────────────────┐
                           │                                     │
                           ▼                                     ▼
                    Credit Decision                         SHAP
                    APPROVE / REJECT                      Explainability
                           │                                     │
                           └──────────────────┬──────────────────┘
                                              │
                                              ▼
                                      FastAPI Model API
                                              │
                           ┌──────────────────┼──────────────────┐
                           │                  │                  │
                           ▼                  ▼                  ▼
                        /health             /score           /explain
                           │                  │                  │
                           └──────────────────┼──────────────────┘
                                              │
                                              ▼
                                      PostgreSQL
                                              │
                              ┌───────────────┼───────────────┐
                              │               │               │
                              ▼               ▼               ▼
                       Applicant Features  Audit Logs   Fairness Results
                                              │
                                              ▼
                                      Streamlit Dashboard
                                              │
                                              ▼
                                         Render Cloud
```

### Production Deployment Architecture

```text
                           GitHub
                              │
                              ▼
                       GitHub Actions
                         CI Pipeline
                              │
                              ▼
                           Render
                              │
              ┌───────────────┼────────────────┐
              │               │                │
              ▼               ▼                ▼
       Streamlit         FastAPI API      PostgreSQL
       Dashboard         Model Serving    Feature Store
              │               │                │
              │               └───────┬────────┘
              │                       │
              └───────────────► Model Inference
                                      │
                           ┌──────────┴──────────┐
                           ▼                     ▼
                     Prediction              SHAP
                     + Decision           Explanation
```

---

## 🌐 Live Deployment

CreditIQ has been deployed as a complete cloud-based ML application using Render.

The production system consists of three main services:

```text
Streamlit Dashboard
        │
        ▼
FastAPI Model API
        │
        ▼
PostgreSQL Feature Store
```

Live links are listed in the [Live Links](#live-links) section below.

---

## Model Development & Comparison

CreditIQ was developed as a supervised binary classification problem using the Home Credit dataset.

The modeling workflow focused on predictive performance, probability quality, explainability, and responsible decision-making rather than optimizing accuracy alone.

### Models Evaluated

| Model | Purpose |
|---|---|
| Logistic Regression | Interpretable baseline model |
| Random Forest | Non-linear tree-based comparison |
| XGBoost | High-performance gradient boosting model |
| Monotonic XGBoost | Final model with constrained feature behavior |

The final model was selected based on the overall validation results and its suitability for a credit-risk decision-support system.

### Final Model

```text
Model:              Monotonic XGBoost
Model Version:      xgboost_monotonic_v1
Number of Features: 80
Decision Threshold: 0.20
```

---

## Probability Calibration & Decision Threshold

Raw model scores were calibrated before being used as production probabilities.

CreditIQ uses **Platt Scaling** to improve the reliability of predicted default probabilities. This is important in credit-risk applications because the output is interpreted as a probability of default rather than only as a class label.

### Calibration Workflow

```text
XGBoost Model
      ↓
Raw Model Output
      ↓
Platt Scaling
      ↓
Calibrated Probability of Default
      ↓
Frozen Decision Threshold
      ↓
Credit Decision
```

---

## Explainability

CreditIQ uses **SHAP (SHapley Additive exPlanations)** to explain individual credit-risk predictions.

Instead of returning only a probability and decision, the system identifies the features that contributed most to increasing or reducing the applicant's predicted risk.

### Applicant-Level Explanation

The `/explain` API endpoint returns:

- Probability of default
- Final decision
- Decision threshold
- Model version
- Top risk-increasing factors
- Top risk-reducing factors
- SHAP contribution values

The explanation separates the factors into two groups:

```text
                    Applicant
                        │
                        ▼
                XGBoost Model
                        │
                        ▼
                  SHAP Analysis
                   /         \
                  /           \
                 ▼             ▼
        Increases Risk    Reduces Risk
```

---

## Fairness & Responsible ML

CreditIQ treats credit-risk prediction as a high-impact machine learning application where predictive performance alone is not sufficient.

The system includes a fairness evaluation layer to identify potential differences in model behavior across protected-attribute proxies.

### Fairness Evaluation

The fairness audit evaluates:

- Demographic parity difference
- Equalized odds difference
- Protected-attribute proxy
- Model version
- Whether mitigation was applied

These results are stored in the PostgreSQL `fairness_audit_results` table for traceability.

### Fairness Audit Flow

```text
Production Model
       ↓
Protected-Attribute Proxy
       ↓
Fairness Evaluation
       ↓
┌──────────────────────────────┐
│ Demographic Parity Difference│
│ Equalized Odds Difference    │
└──────────────┬───────────────┘
               ↓
        Audit Result
               ↓
          PostgreSQL
```

---

## Data & Database

CreditIQ uses the Home Credit dataset as its primary source for applicant-level credit-risk modeling.

The raw data is transformed through data preparation and feature engineering before being stored as model-ready applicant features in PostgreSQL.

### Dataset

```text
307,511 applicants
80 production model features
```

---

## API & Model Serving

CreditIQ exposes the production machine learning model through a FastAPI REST API.

The API separates the model inference layer from the user interface, allowing the same prediction service to be accessed by the Streamlit dashboard or external applications.

### API Architecture

```text
Client / Dashboard
        ↓
     FastAPI
        ↓
Request Validation
        ↓
PostgreSQL Feature Store
        ↓
Feature Retrieval
        ↓
XGBoost Model
        ↓
Probability of Default
        ↓
Decision Threshold
        ↓
Response
```

---

## Streamlit Dashboard

CreditIQ includes an interactive, full-width Streamlit dashboard for applicant-level credit-risk assessment.

The dashboard communicates with the deployed FastAPI service rather than running the model directly in the browser. It shows the probability of default on a risk spectrum, the decision at the approved cutoff, a factor chart of what drove the score, a what-if cutoff slider, a session history of recent assessments, and a downloadable JSON report.

### Dashboard Flow

```text
User
  ↓
Enter Applicant ID
  ↓
Streamlit Dashboard
  ↓
FastAPI
  ↓
PostgreSQL Feature Store
  ↓
XGBoost Model
  ↓
Probability + Decision
  ↓
SHAP Explanation
  ↓
Risk Spectrum + Factor Chart
```

---

## Docker & Containerization

CreditIQ is containerized using Docker to make the application reproducible across development and deployment environments.

The project separates the API and dashboard into independent Docker images while using Docker Compose for local orchestration.

### Docker Services

```text
Docker Compose
      │
      ├── PostgreSQL
      │
      ├── FastAPI API
      │
      ├── Streamlit Dashboard
      │
      └── MLflow
```

---

## CI/CD — GitHub Actions

CreditIQ uses GitHub Actions to automatically validate changes before they are considered ready for deployment.

The CI pipeline helps ensure that application tests pass and that the production Docker images can be built successfully.

### CI Pipeline

```text
Git Push / Pull Request
          ↓
   Checkout Repository
          ↓
   Setup Python 3.11
          ↓
   Install Dependencies
          ↓
      Run pytest
          ↓
   Build API Docker Image
          ↓
Build Dashboard Docker Image
          ↓
      CI Complete
```

---

## MLflow & Model Tracking

CreditIQ uses **MLflow** to support experiment tracking and model lifecycle management.

MLflow helps connect model development with the final production model by keeping track of experiments, model versions, and associated evaluation information.

### MLflow Workflow

```text
Model Experiments
        ↓
Performance Evaluation
        ↓
Model Selection
        ↓
Model Version
        ↓
Production Deployment
```

---

## Production Validation

CreditIQ was validated after deployment by testing the complete production workflow from the public dashboard and API through the production PostgreSQL feature store and machine learning model.

### Production Health Check

The deployed API successfully returned:

```json
{
  "status": "healthy",
  "service": "CreditIQ API",
  "model_loaded": true,
  "database_connected": true
}
```

---

## Security & Configuration

CreditIQ keeps environment-specific configuration outside the application source code.

Sensitive values such as database credentials are supplied through environment variables rather than being hard-coded into the application.

### Environment Variables

The project uses configuration variables including:

```text
DB_HOST
DB_PORT
DB_NAME
DB_USER
DB_PASSWORD
API_URL
MLFLOW_TRACKING_URI
```

---

## Project Structure

```text
CreditIQ/
│
├── api/
│   ├── main.py
│   ├── dependencies.py
│   │
│   ├── routes/
│   │   ├── health.py
│   │   ├── score.py
│   │   └── explain.py
│   │
│   └── schemas/
│       ├── request.py
│       └── response.py
│
├── configs/
│   ├── fairness_config.yaml
│   ├── logging_config.yaml
│   ├── model_config.yaml
│   └── threshold.yaml
│
├── dashboard/
│   └── streamlit_app.py
│
├── database/
│   ├── schema.sql
│   └── test_join.sql
│
├── docker/
│   ├── Dockerfile.api
│   └── Dockerfile.dashboard
│
├── docs/
│   ├── architecture.md
│   ├── data_dictionary.md
│   └── model_card.md
│
├── models/
│   ├── baseline_logistic_regression.joblib
│   ├── platt_calibrator.joblib
│   ├── xgboost_monotonic.joblib
│   ├── xgboost_unconstrained.joblib
│   └── feature_store/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_leakage_investigation.ipynb
│   └── 03_woe_iv_exploration.ipynb
│
├── reports/
│   ├── figures/
│   └── metrics/
│
├── scripts/
│   ├── run_full_pipeline.sh
│   ├── run_local_stack.sh
│   ├── seed_db.py
│   └── seed_docker_db.py
│
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── explainability/
│   ├── fairness/
│   ├── features/
│   ├── mlops/
│   ├── models/
│   ├── synthetic/
│   └── utils/
│
├── tests/
│   ├── test_api.py
│   ├── test_etl.py
│   ├── test_fairness_audit.py
│   ├── test_monotonic_constraints.py
│   └── test_woe_encoder.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/vaishnavi-c12488/CreditIQ.git
cd CreditIQ
```

### 2. Create a Virtual Environment and Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your PostgreSQL credentials and `API_URL`.

### 4. Start the Full Stack

```bash
docker compose up --build
```

Then open the API documentation at <http://localhost:8000/docs>.

---

## Testing

CreditIQ includes automated tests covering important parts of the application and ML pipeline.

### Test Suite

The project includes tests for:

- API behavior
- ETL processing
- Fairness auditing
- Monotonic constraints
- WOE encoding

Run the complete test suite with:

```bash
pytest -q
```

---

## Future Improvements

CreditIQ currently provides the complete foundation for a production-oriented credit-risk system. Future improvements could include:

- Automated model retraining pipelines
- Continuous model-drift monitoring
- Automated data-quality monitoring
- Automated fairness monitoring
- Model performance monitoring in production
- Authentication and role-based access control
- More advanced model registry workflows
- Automated cloud deployment pipelines
- Production alerting and observability
- Scalable infrastructure for larger workloads
- Human-in-the-loop review workflows

---

## Technology Stack

| Category | Technology |
|---|---|
| Language | Python 3.11 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Fairness | Fairlearn |
| Calibration | Platt Scaling |
| Database | PostgreSQL |
| API | FastAPI |
| Dashboard | Streamlit |
| Experiment Tracking | MLflow |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Cloud | Render |
| API Documentation | Swagger / OpenAPI |
| Version Control | Git / GitHub |

---

## Key Takeaways

CreditIQ demonstrates how a machine learning model can be taken from experimentation to a working production application.

The project combines:

```text
Data Engineering
       +
Feature Engineering
       +
Machine Learning
       +
Probability Calibration
       +
Fairness Evaluation
       +
Explainable AI
       +
PostgreSQL
       +
FastAPI
       +
Streamlit
       +
Docker
       +
CI/CD
       +
Cloud Deployment
```

---

## Production Checklist

```text
✓ 307,511 applicants in production feature store
✓ 80 model-ready features
✓ Monotonic XGBoost
✓ ROC-AUC: 0.7610
✓ PR-AUC: 0.2266
✓ Accuracy: 91.56%
✓ Precision: 31.20%
✓ Recall: 34.17%
✓ F1 Score: 32.71%
✓ Calibration Error: 0.0006
✓ Frozen decision threshold: 0.20
✓ Platt probability calibration
✓ SHAP TreeExplainer
✓ Fairness evaluation
✓ PostgreSQL feature store
✓ FastAPI model serving
✓ Streamlit risk dashboard
✓ MLflow tracking
✓ Dockerized services
✓ GitHub Actions CI
✓ Render cloud deployment
✓ Production /health validation
✓ Production /score validation
✓ Production /explain validation
✓ End-to-end production testing
```

---

## Live Links

| Resource | Link |
|---|---|
| 🌐 CreditIQ Dashboard | https://creditiq-dashboard-vygi.onrender.com |
| ⚡ CreditIQ API | https://creditiq-a2iw.onrender.com |
| 📚 Swagger API Documentation | https://creditiq-a2iw.onrender.com/docs |
| 💻 GitHub Repository | https://github.com/vaishnavi-c12488/CreditIQ |

---

## Author

**Vaishnavi Chavan**

B.Sc. Data Science
Symbiosis Skills and Professional University, Pune

Interested in:

- Data Science
- Machine Learning
- ML Engineering
- Explainable AI
- Production ML Systems
