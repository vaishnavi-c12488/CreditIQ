SELECT
    "SK_ID_CURR",

    COUNT(*) AS credit_card_record_count,

    AVG("MONTHS_BALANCE") AS credit_card_avg_months_balance,

    MIN("MONTHS_BALANCE") AS credit_card_oldest_month,

    AVG("AMT_BALANCE") AS credit_card_avg_balance,

    MAX("AMT_BALANCE") AS credit_card_max_balance,

    AVG("AMT_CREDIT_LIMIT_ACTUAL") AS credit_card_avg_limit,

    MAX("AMT_CREDIT_LIMIT_ACTUAL") AS credit_card_max_limit,

    AVG("AMT_DRAWINGS_CURRENT") AS credit_card_avg_drawings,

    SUM("AMT_DRAWINGS_CURRENT") AS credit_card_total_drawings,

    AVG("AMT_PAYMENT_CURRENT") AS credit_card_avg_payment,

    SUM("AMT_PAYMENT_CURRENT") AS credit_card_total_payment,

    AVG("AMT_TOTAL_RECEIVABLE") AS credit_card_avg_receivable,

    AVG("SK_DPD") AS credit_card_avg_dpd,

    MAX("SK_DPD") AS credit_card_max_dpd,

    AVG("SK_DPD_DEF") AS credit_card_avg_dpd_def,

    MAX("SK_DPD_DEF") AS credit_card_max_dpd_def

FROM credit_card_balance

GROUP BY "SK_ID_CURR";