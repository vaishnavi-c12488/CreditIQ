SELECT
    "SK_ID_CURR",

    COUNT(*) AS bureau_loan_count,

    AVG("AMT_CREDIT_SUM") AS bureau_avg_credit,

    SUM("AMT_CREDIT_SUM") AS bureau_total_credit,

    AVG("AMT_CREDIT_SUM_DEBT") AS bureau_avg_debt,

    SUM("AMT_CREDIT_SUM_DEBT") AS bureau_total_debt,

    AVG("AMT_CREDIT_SUM_LIMIT") AS bureau_avg_credit_limit,

    SUM("AMT_CREDIT_SUM_LIMIT") AS bureau_total_credit_limit,

    AVG("AMT_ANNUITY") AS bureau_avg_annuity,

    MAX("DAYS_CREDIT") AS bureau_most_recent_days,

    MIN("DAYS_CREDIT") AS bureau_oldest_days

FROM bureau

GROUP BY "SK_ID_CURR";