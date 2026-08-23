-- ============================================================
-- CreditIQ Database Schema
-- ============================================================
-- Application-owned tables.
--
-- The seven original Home Credit tables are NOT defined here.
-- They are loaded separately as raw-mirror tables.
-- ============================================================


-- ============================================================
-- 1. Applicant Features
-- ============================================================
-- Final model-ready feature table.
-- One row = one applicant (SK_ID_CURR).
-- Produced later by ETL + feature engineering + WOE/IV.

CREATE TABLE IF NOT EXISTS applicant_features (
    sk_id_curr INTEGER PRIMARY KEY,

    feature_snapshot_version VARCHAR(20) NOT NULL,

    income_woe FLOAT,
    credit_amount_woe FLOAT,
    annuity_woe FLOAT,
    ext_source_1_woe FLOAT,
    ext_source_2_woe FLOAT,
    ext_source_3_woe FLOAT,

    days_employed_clean FLOAT,
    debt_to_income_ratio FLOAT,

    prior_bureau_credit_count INTEGER,
    prior_application_refused_count INTEGER,
    installment_ontime_ratio FLOAT,

    -- Additional engineered features will be added
    -- as the feature-engineering pipeline is developed.

    target INTEGER,

    created_at TIMESTAMP DEFAULT NOW()
);


-- ============================================================
-- 2. Scoring Audit Log
-- ============================================================
-- Stores every production scoring decision.
-- Used for traceability and audit/compliance.

CREATE TABLE IF NOT EXISTS scoring_audit_log (
    id SERIAL PRIMARY KEY,

    sk_id_curr INTEGER,

    request_payload JSONB NOT NULL,

    predicted_probability FLOAT NOT NULL,

    decision VARCHAR(20) NOT NULL,

    top_reason_codes JSONB NOT NULL,

    model_version VARCHAR(50) NOT NULL,

    scored_at TIMESTAMP DEFAULT NOW()
);


-- ============================================================
-- 3. Fairness Audit Results
-- ============================================================
-- Stores fairness metrics for each model version.

CREATE TABLE IF NOT EXISTS fairness_audit_results (
    id SERIAL PRIMARY KEY,

    model_version VARCHAR(50) NOT NULL,

    protected_attribute_proxy VARCHAR(50) NOT NULL,

    demographic_parity_difference FLOAT NOT NULL,

    equalized_odds_difference FLOAT NOT NULL,

    mitigation_applied BOOLEAN NOT NULL,

    audited_at TIMESTAMP DEFAULT NOW()
);


-- ============================================================
-- Indexes
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_audit_log_sk_id
ON scoring_audit_log(sk_id_curr);


CREATE INDEX IF NOT EXISTS idx_audit_log_scored_at
ON scoring_audit_log(scored_at);


CREATE INDEX IF NOT EXISTS idx_fairness_model_version
ON fairness_audit_results(model_version);