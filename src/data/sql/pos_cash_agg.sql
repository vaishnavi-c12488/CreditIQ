SELECT
    "SK_ID_CURR",

    COUNT(*) AS pos_cash_record_count,

    AVG("MONTHS_BALANCE") AS pos_cash_avg_months_balance,

    MIN("MONTHS_BALANCE") AS pos_cash_oldest_month,

    AVG("CNT_INSTALMENT") AS pos_cash_avg_installments,

    AVG("CNT_INSTALMENT_FUTURE") AS pos_cash_avg_future_installments,

    AVG("SK_DPD") AS pos_cash_avg_dpd,

    MAX("SK_DPD") AS pos_cash_max_dpd,

    AVG("SK_DPD_DEF") AS pos_cash_avg_dpd_def,

    MAX("SK_DPD_DEF") AS pos_cash_max_dpd_def,

    COUNT(*) FILTER (
        WHERE "NAME_CONTRACT_STATUS" = 'Active'
    ) AS pos_cash_active_count,

    COUNT(*) FILTER (
        WHERE "NAME_CONTRACT_STATUS" = 'Completed'
    ) AS pos_cash_completed_count

FROM POS_CASH_balance

GROUP BY "SK_ID_CURR";