# CreditIQ Model Card

## 1. Model Overview

**Project:** CreditIQ  
**Task:** Credit default risk prediction  
**Model Type:** Monotonic Gradient-Boosted Decision Trees  
**Final Model:** Monotonic XGBoost  
**Probability Calibration:** Platt scaling  
**Decision Threshold:** 0.20  

CreditIQ is a credit-risk prediction system designed to estimate the
probability of loan default and support risk-based credit decisions.

The final pipeline combines WOE/IV-based feature engineering, a
monotonic-constrained XGBoost model, probability calibration, and a
frozen decision threshold selected using validation data.

The system was evaluated using threshold-independent ranking metrics,
calibration metrics, threshold-dependent operating metrics,
explainability analysis, and a fairness audit.

## 2. Intended Use

CreditIQ is intended for:

- Credit-risk assessment and applicant risk ranking.
- Supporting risk-based lending decisions.
- Identifying applicants with elevated estimated default risk.
- Providing interpretable model outputs for analysis and review.
- Evaluating model behavior across demographic proxy groups.

The model is intended as a decision-support system rather than an
autonomous replacement for human credit-risk governance.

### Out-of-Scope Uses

CreditIQ should not be treated as:

- A guarantee that an applicant will or will not default.
- A standalone basis for legally binding lending decisions.
- Proof of legal or ethical fairness across all protected attributes.
- A substitute for institutional credit policy, regulatory review,
  or human oversight.

  ## 3. Dataset

CreditIQ uses the Home Credit Default Risk dataset.

The project works with multiple related sources covering:

- Loan application information
- Previous credit history
- Previous Home Credit applications
- Repayment behavior
- POS/cash-loan history
- Credit-card history

The raw data is preserved separately from the processed datasets.

Feature engineering and model preparation are performed on processed
data before the modeling stage.

## 4. Data Splitting and Evaluation

CreditIQ maintains separate training, validation, and test datasets.

The validation data is used for:

- Model comparison
- Calibration development
- Threshold analysis
- Decision-threshold selection
- Fairness analysis

The final test dataset is kept separate from model development and is
used only for the final evaluation.

The decision threshold was selected using validation data and frozen
before the final test evaluation.

The final test evaluation was then performed using the frozen threshold
of `0.20`.

This separation prevents the final test results from influencing model
selection or threshold tuning.

## 5. Feature Engineering

CreditIQ applies feature engineering to transform the original
application and historical credit information into predictive,
model-ready variables.

The feature-engineering stage includes derived financial,
employment, repayment, and credit-history indicators.

Examples of engineered features used by the final model include:

- `CREDIT_UTILIZATION`
- `ANNUITY_TO_CREDIT`
- `YEARS_EMPLOYED`
- `installment_ontime_ratio`
- `installment_total_paid`
- `installment_avg_paid`
- `previous_avg_credit_to_application_ratio`
- `bureau_most_recent_days`

The project preserves the distinction between raw variables and
processed model features.

## 6. Weight of Evidence (WOE) and Information Value (IV)

WOE encoding is used to transform model features into risk-oriented
representations.

The feature-selection process uses Information Value (IV) to identify
variables with useful predictive information while reducing the number
of features presented to the final model.

The final model uses **80 selected features**.

WOE transformation is applied consistently to the modeling datasets so
that the training, validation, and final test datasets use the same
feature representation.

The resulting WOE features are used as inputs to the downstream
Logistic Regression and XGBoost models.

### WOE/IV Objectives

The WOE/IV stage is designed to:

- Transform variables into risk-oriented representations.
- Reduce the influence of weak or uninformative variables.
- Support model interpretability.
- Provide a consistent feature representation across datasets.
- Support monotonic risk relationships where appropriate.

## 7. Model Development

Three supervised learning approaches were evaluated during model
development:

1. Logistic Regression — baseline model
2. XGBoost — unconstrained model
3. XGBoost — monotonic-constrained model

The models were evaluated on the validation dataset using metrics
appropriate for an imbalanced credit-default prediction problem.

### Validation Model Comparison

| Model | ROC-AUC | PR-AUC | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7531 | 0.2493 | 0.7694 | 0.2721 |
| XGBoost — Unconstrained | 0.7653 | 0.2755 | 0.7669 | 0.2816 |
| **XGBoost — Monotonic** | **0.7648** | **0.2714** | **0.7701** | **0.2813** |

## 8. Final Model Selection

The **Monotonic XGBoost** model was selected as the final model.

The unconstrained XGBoost model achieved a marginally higher validation
ROC-AUC (`0.7653` vs. `0.7648`). However, the difference was extremely
small.

The monotonic model was preferred because it incorporates explicit
monotonic risk constraints on:

- `EXT_SOURCE_1`
- `EXT_SOURCE_2`
- `EXT_SOURCE_3`

The constraints enforce the expected direction of the relationship
between these external-source features and predicted default risk.

The final model therefore prioritizes a combination of predictive
performance and controlled risk behavior rather than selecting a model
solely on the highest ROC-AUC.

The selected model was subsequently used for probability calibration,
threshold analysis, explainability, and final evaluation.

## 9. Probability Calibration

The selected Monotonic XGBoost model produces raw probability estimates
that require calibration before being used as credit-risk probabilities.

CreditIQ applies **Platt scaling** to calibrate the model outputs.

The calibration pipeline is:

Raw XGBoost probability
→ Log-odds transformation
→ Platt calibrator
→ Calibrated probability

The original validation dataset was divided into separate calibration
and evaluation subsets:

- Calibration set: 23,063 observations
- Calibration evaluation set: 23,064 observations

The Platt calibrator was fitted only on the calibration subset and
evaluated on the separate evaluation subset.

### Calibration Results

| Metric | Before Calibration | After Calibration |
|---|---:|---:|
| ROC-AUC | 0.7630 | 0.7630 |
| PR-AUC | 0.2747 | 0.2747 |
| KS | 0.3883 | 0.3883 |
| Calibration Error | 0.3624 | **0.0039** |

Calibration substantially reduced the calibration error from **0.3624
to 0.0039**, while leaving the ranking metrics unchanged.

This is expected because calibration adjusts the probability scale rather
than changing the underlying ranking of applicants.

The fitted calibrator is stored as:

`models/platt_calibrator.joblib`

The corresponding evaluation results are stored in:

`reports/metrics/calibration_results.json`

The calibration comparison visualization is stored in:

`reports/figures/calibration_curve_comparison.png`

## 10. Decision Threshold Selection

The model produces continuous calibrated default probabilities. A
decision threshold is therefore required to convert these probabilities
into default/approval decisions.

The default classification threshold of `0.50` was evaluated but was
not suitable for the CreditIQ decision objective.

At a threshold of `0.50`, the final test evaluation produced high
accuracy but very low default recall. This demonstrated that accuracy
alone could be misleading because the dataset contains substantially
more non-default observations than defaults.

### Validation-Based Threshold Analysis

A threshold sweep was performed using the validation dataset only.

The analysis considered:

- Precision
- Recall
- F1
- False Positive Rate (FPR)
- Approval rate
- Threshold-level KS separation

The selected operating threshold was:

**Decision threshold: `0.20`**

At this threshold on the validation evaluation:

| Metric | Validation |
|---|---:|
| Precision | 29.98% |
| Recall | 37.67% |
| F1 | 0.3339 |
| False Positive Rate | 9.12% |
| Approval Rate | 88.20% |

The threshold was selected before the final test evaluation and then
frozen.

The decision policy is stored in:

`configs/threshold.yaml`

The complete threshold sweep is stored in:

`reports/metrics/threshold_analysis.json`

### Test-Set Isolation

The final test dataset was not used to select the decision threshold.

After threshold `0.20` was frozen, the test dataset was evaluated using
the finalized model, calibration pipeline, and frozen threshold.

This prevents the final test results from influencing the operating-point
selection.

## 11. Final Test Performance

After model selection, calibration, and decision-threshold selection were
completed, the untouched test dataset was evaluated exactly once using
the finalized pipeline:

Monotonic XGBoost
→ Log-odds transformation
→ Platt calibration
→ Frozen threshold `0.20`

### Final Test Results

| Metric | Test Result |
|---|---:|
| ROC-AUC | **0.7630** |
| PR-AUC | **0.2826** |
| KS-statistic | **0.3942** |
| Calibration Error | **0.0096** |
| Precision | **31.38%** |
| Recall | **34.17%** |
| F1 | **0.3271** |
| False Positive Rate | **8.44%** |
| Approval Rate | **88.95%** |
| Decline Rate | **11.05%** |
| Accuracy | 85.74% |

### Confusion Matrix

At the frozen threshold of `0.20`:

| | Predicted Non-Default | Predicted Default |
|---|---:|---:|
| **Actual Non-Default** | 37,950 | 3,497 |
| **Actual Default** | 3,081 | 1,599 |

The model correctly identified 1,599 of the 4,680 actual default cases
in the final test dataset.

### Accuracy Interpretation

Test accuracy was **85.74%** and is reported for transparency, but it is
not the primary model-selection metric.

Because the dataset is imbalanced toward non-default observations,
accuracy can give an incomplete picture of credit-risk model usefulness.
For example, a model that predicts non-default for every applicant can
achieve high accuracy while identifying no defaults.

CreditIQ therefore emphasizes:

- ROC-AUC for overall ranking discrimination.
- PR-AUC for performance under class imbalance.
- KS-statistic for separation between default and non-default populations.
- Calibration Error for probability reliability.
- Recall, precision, F1, and FPR at the frozen operating threshold.

The final test results are stored in:

`reports/metrics/final_test_results.json`

The test dataset was not used during model selection, calibration, or
threshold selection.

## 12. Model Explainability — SHAP

CreditIQ uses SHAP (SHapley Additive exPlanations) to analyze how
individual features contribute to the Monotonic XGBoost model's
predictions.

SHAP analysis was performed on **5,000 validation observations** using
a TreeExplainer.

The analysis provides:

- Global feature importance using mean absolute SHAP values.
- Distribution of feature effects using a SHAP beeswarm plot.
- Feature-specific dependence plots for the most influential variables.

### Top Features by Mean Absolute SHAP

| Rank | Feature | Mean Absolute SHAP |
|---:|---|---:|
| 1 | `EXT_SOURCE_2` | 0.335811 |
| 2 | `EXT_SOURCE_3` | 0.311373 |
| 3 | `EXT_SOURCE_1` | 0.171788 |
| 4 | `installment_ontime_ratio` | 0.139023 |
| 5 | `previous_avg_credit_to_application_ratio` | 0.135827 |
| 6 | `ANNUITY_TO_CREDIT` | 0.120554 |
| 7 | `AMT_GOODS_PRICE` | 0.109586 |
| 8 | `CODE_GENDER` | 0.108759 |
| 9 | `OWN_CAR_AGE` | 0.092718 |
| 10 | `NAME_EDUCATION_TYPE` | 0.081715 |

The three `EXT_SOURCE` variables are the strongest contributors to
global model predictions, followed by repayment behavior and
credit/application-related features.

### SHAP Artifacts

The explainability outputs are stored under:

`reports/metrics/`

- `shap_feature_importance.csv`
- `shap_summary.json`

Visualizations are stored under:

`reports/figures/`

- `shap_summary_bar.png`
- `shap_summary_beeswarm.png`
- `shap_dependence_EXT_SOURCE_1.png`
- `shap_dependence_EXT_SOURCE_2.png`
- `shap_dependence_EXT_SOURCE_3.png`
- `shap_dependence_installment_ontime_ratio.png`
- `shap_dependence_previous_avg_credit_to_application_ratio.png`

SHAP is used for model interpretation and diagnostic analysis. It does
not change the trained model or the final decision threshold.

## 13. Fairness Audit

CreditIQ includes a fairness audit using Fairlearn to evaluate whether
model decisions differ across age groups.

Because a directly protected demographic attribute is not used in the
available dataset, age groups are derived from `DAYS_BIRTH` and are
treated as a proxy attribute.

### Age Groups

The audit evaluates five groups:

- `<25`
- `25-34`
- `35-44`
- `45-54`
- `55+`

### Fairness Before Mitigation

Using the finalized model and frozen decision threshold of `0.20`:

| Metric | Before Mitigation |
|---|---:|
| Demographic Parity Difference | 0.2036 |
| Equalized Odds Difference | 0.3787 |

The audit identified measurable disparity across the age-proxy groups.

### Fairlearn Mitigation

Fairlearn's `ThresholdOptimizer` was evaluated using an
**equalized-odds constraint**.

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Demographic Parity Difference | 0.2036 | **0.0178** | -0.1858 |
| Equalized Odds Difference | 0.3787 | **0.0220** | -0.3567 |
| Precision | 29.73% | 19.26% | -10.46 pp |
| Recall | 37.21% | **64.01%** | +26.80 pp |
| F1 | 0.3305 | 0.2961 | -0.0344 |
| False Positive Rate | 9.12% | 27.81% | +18.69 pp |
| Approval Rate | 88.24% | 68.79% | -19.46 pp |

The mitigation substantially reduced both measured fairness disparities,
with Equalized Odds Difference decreasing from `0.3787` to `0.0220`.

However, the mitigation also produced a material change in decision
behavior, including a higher false-positive rate, lower precision,
lower F1, and substantially lower approval rate.

### Mitigation Decision

The Fairlearn-mitigated policy is therefore **not automatically adopted
as the production CreditIQ decision policy**.

It is retained as a documented fairness-mitigation analysis so that the
trade-off between fairness and predictive/operating performance remains
visible.

The finalized CreditIQ decision pipeline remains:

`Monotonic XGBoost → Platt calibration → frozen threshold 0.20`

The fairness audit is stored in:

`reports/metrics/fairness_audit.json`

### Fairness Limitation

The age groups used in this audit are derived from `DAYS_BIRTH` and are
therefore a proxy attribute.

This audit does not establish complete legal, regulatory, or ethical
fairness across all protected attributes.

Fairness results should therefore be interpreted as an empirical audit
of the evaluated age-proxy groups rather than a universal fairness
guarantee.

## 14. Monotonic Constraint Verification

The final XGBoost model applies monotonic constraints to:

- `EXT_SOURCE_1`
- `EXT_SOURCE_2`
- `EXT_SOURCE_3`

The constraints enforce the expected direction of the relationship between
these features and predicted default risk.

The constraints were independently verified using a validation-data
monotonicity sweep.

For each constrained feature, values were varied across a realistic
validation-data range while the remaining applicant characteristics were
held constant.

The resulting predicted default probabilities were checked for
violations of the specified monotonic direction.

A zero-violation result is required for the constraint verification to
pass.

The verification script is:

`tests/test_monotonic_constraints.py`

### Monotonicity Verification Result

The monotonicity sweep produced zero violations for all three constrained
features:

| Feature | Violations | Result |
|---|---:|---|
| `EXT_SOURCE_1` | 0 | PASS |
| `EXT_SOURCE_2` | 0 | PASS |
| `EXT_SOURCE_3` | 0 | PASS |

**Result: All monotonic constraints were satisfied.**

## 15. Automated Testing

CreditIQ includes automated tests covering important parts of the
modeling pipeline.

The complete test suite currently passes:

```text
10 passed
```

## 16. Development Issues and Resolutions

Several implementation issues were identified during development and
resolved through validation and testing.

### 16.1 Misleading Performance at the Default Threshold

**Issue:**  
Using the conventional `0.50` classification threshold produced high
accuracy but very low default recall.

**Investigation:**  
The dataset contains substantially more non-default observations than
defaults. Consequently, accuracy alone did not adequately represent the
model's ability to identify default cases.

**Resolution:**  
A validation-only threshold analysis was introduced. The final operating
threshold was selected using a false-positive-rate objective and frozen
at `0.20` before the final test evaluation.

**Verification:**  
The final test set was evaluated only after the threshold was frozen.

---

### 16.2 Age Proxy Initially Collapsed to a Single Group

**Issue:**  
The initial fairness audit produced only one age group because the
`DAYS_BIRTH` values were interpreted incorrectly when creating the age
proxy.

**Investigation:**  
The original validation dataset was inspected directly and confirmed that
`DAYS_BIRTH` contains negative day values representing age relative to
the application date.

**Resolution:**  
The age calculation was corrected using the absolute value of
`DAYS_BIRTH`, converted to years, and grouped into five age ranges.

**Verification:**  
The final fairness audit successfully identified five age groups:

- `<25`
- `25-34`
- `35-44`
- `45-54`
- `55+`

---

### 16.3 Fairness Mitigation Trade-off

**Issue:**  
Fairlearn mitigation substantially reduced measured age-proxy disparity,
but also changed the model's operating characteristics.

**Investigation:**  
Performance was measured before and after Fairlearn's
Equalized-odds-constrained threshold optimization.

**Resolution:**  
The mitigated policy was retained as a fairness-analysis result rather
than automatically replacing the finalized CreditIQ decision policy.

**Verification:**  
Equalized Odds Difference decreased from `0.3787` to `0.0220`, while
the false-positive rate increased from `9.12%` to `27.81%`.

The trade-off is documented rather than hidden.

---

### 16.4 Test-Set Isolation

**Issue:**  
Model selection, calibration, and threshold selection must not be
influenced by final test results.

**Resolution:**  
The development workflow separates validation-based model development
from the final test evaluation.

The decision threshold was frozen in:

`configs/threshold.yaml`

before the final test evaluation.

**Verification:**  
The final test dataset was evaluated only after model selection,
calibration, and threshold selection were completed.

---

### 16.5 Model Selection Trade-off

**Issue:**  
Unconstrained XGBoost achieved a marginally higher validation ROC-AUC
than the monotonic model.

**Investigation:**  
The difference was:

`0.7653 − 0.7648 = 0.0005`

**Resolution:**  
Monotonic XGBoost was selected because it provides explicit monotonic
constraints on the three `EXT_SOURCE` features while maintaining
essentially equivalent ranking performance.

**Verification:**  
The monotonic constraints were independently tested using a validation
data sweep.

## 17. Limitations

The CreditIQ model has several important limitations.

### Data Limitations

The model is trained and evaluated using the available Home Credit
Default Risk dataset. Model performance may differ when applied to
different populations, lending products, geographic regions, or future
economic conditions.

### Class Imbalance

Default cases represent a minority of observations. Consequently,
accuracy is not considered an adequate standalone measure of model
quality.

### Fairness Limitations

The fairness analysis uses age groups derived from `DAYS_BIRTH` as a
proxy attribute. It does not evaluate every legally protected attribute
and therefore cannot establish complete legal or ethical fairness.

### Probability and Decision Limitations

Calibrated probabilities and the frozen threshold are derived from the
available validation data. Changes in population characteristics,
default rates, or business policy may require recalibration or
reassessment of the operating threshold.

### Model Explainability Limitations

SHAP provides feature-attribution information for model predictions, but
feature importance should not automatically be interpreted as causal
relationships.

### Operational Limitations

The current project demonstrates the modeling, evaluation,
explainability, fairness, and validation pipeline. Production deployment
would additionally require monitoring, data-quality controls, access
controls, model governance, drift detection, and periodic validation.

---

## 18. Reproducibility

The project stores important model-development artifacts separately from
source code and raw data.

Key reproducibility artifacts include:

- Final trained model:
  `models/xgboost_monotonic.joblib`
- Platt calibrator:
  `models/platt_calibrator.joblib`
- Frozen decision policy:
  `configs/threshold.yaml`
- Final test metrics:
  `reports/metrics/final_test_results.json`
- Threshold analysis:
  `reports/metrics/threshold_analysis.json`
- Calibration results:
  `reports/metrics/calibration_results.json`
- SHAP results:
  `reports/metrics/shap_summary.json`
- Fairness results:
  `reports/metrics/fairness_audit.json`

````markdown
The automated test suite can be executed with:

```bash
pytest -q

The current test suite completes successfully with:

```text
10 passed
```
## 19. Project Structure

```text
CreditIQ/
│
├── configs/
│   └── threshold.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── baseline_logistic_regression.joblib
│   ├── xgboost_unconstrained.joblib
│   ├── xgboost_monotonic.joblib
│   └── platt_calibrator.joblib
│
├── reports/
│   ├── figures/
│   └── metrics/
│
├── src/
│   ├── models/
│   ├── explainability/
│   └── fairness/
│
├── tests/
│
├── docs/
│   └── model_card.md
│
└── README.md
```

The project separates data, configuration, trained models, evaluation
artifacts, source code, tests, and documentation to support
reproducibility and maintainability.

---

## 20. Final Model Pipeline

```text
Raw Home Credit Data
        │
        ▼
Data Processing & Feature Engineering
        │
        ▼
WOE / IV Transformation
        │
        ▼
80 Selected Features
        │
        ▼
Model Comparison
        │
        ├── Logistic Regression
        ├── Unconstrained XGBoost
        └── Monotonic XGBoost
                    │
                    ▼
          Selected Final Model
          Monotonic XGBoost
                    │
                    ▼
             Log-Odds Transform
                    │
                    ▼
             Platt Calibration
                    │
                    ▼
          Calibrated Default Risk
                    │
                    ▼
          Frozen Threshold = 0.20
                    │
                    ▼
            Credit Decision
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
   SHAP Explainability   Fairness Audit
```

The final test dataset remains outside the development pipeline until
the final evaluation stage.

---

## 21. Final Model Summary

CreditIQ's final model is a monotonic-constrained XGBoost classifier
with Platt probability calibration and a frozen decision threshold of
`0.20`.

The final test evaluation produced:

- **ROC-AUC:** 0.7630
- **PR-AUC:** 0.2826
- **KS:** 0.3942
- **Calibration Error:** 0.0096
- **Precision:** 31.38%
- **Recall:** 34.17%
- **F1:** 0.3271
- **False Positive Rate:** 8.44%
- **Approval Rate:** 88.95%
- **Accuracy:** 85.74%

Accuracy is reported for transparency but is not treated as the primary
model-quality metric.

The model was additionally evaluated using SHAP explainability,
monotonicity verification, and a Fairlearn fairness audit.

The fairness mitigation analysis reduced Equalized Odds Difference from
`0.3787` to `0.0220`, but also introduced a significant operating-point
trade-off. Therefore, the mitigated policy was not automatically adopted
as the final CreditIQ decision policy.

---

## 22. Conclusion

CreditIQ provides an end-to-end credit-risk modeling workflow covering
data preparation, feature engineering, WOE/IV transformation, model
comparison, monotonic modeling, probability calibration, threshold
selection, final test evaluation, explainability, fairness analysis, and
automated testing.

The final pipeline prioritizes appropriate credit-risk evaluation
metrics and preserves a strict separation between model development and
final test evaluation.

The resulting model and supporting artifacts provide a reproducible
foundation for further deployment, monitoring, validation, and
credit-risk governance.