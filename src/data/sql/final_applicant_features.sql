WITH bureau_agg AS (

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
    GROUP BY "SK_ID_CURR"
),

previous_application_agg AS (

    SELECT
        "SK_ID_CURR",
        COUNT(*) AS previous_application_count,
        COUNT(*) FILTER (
            WHERE "NAME_CONTRACT_STATUS" = 'Approved'
        ) AS previous_approved_count,
        COUNT(*) FILTER (
            WHERE "NAME_CONTRACT_STATUS" = 'Refused'
        ) AS previous_refused_count,
        AVG("AMT_CREDIT") AS previous_avg_credit,
        SUM("AMT_CREDIT") AS previous_total_credit,
        AVG("AMT_APPLICATION") AS previous_avg_application,
        AVG(
            CASE
                WHEN "AMT_APPLICATION" > 0
                THEN "AMT_CREDIT" / "AMT_APPLICATION"
            END
        ) AS previous_avg_credit_to_application_ratio
    FROM previous_application
    GROUP BY "SK_ID_CURR"
),

installments_agg AS (

    SELECT
        "SK_ID_CURR",
        COUNT(*) AS installment_payment_count,
        AVG("AMT_INSTALMENT") AS installment_avg_due,
        AVG("AMT_PAYMENT") AS installment_avg_paid,
        SUM("AMT_INSTALMENT") AS installment_total_due,
        SUM("AMT_PAYMENT") AS installment_total_paid,
        AVG(
            CASE
                WHEN "AMT_INSTALMENT" > 0
                THEN "AMT_PAYMENT" / "AMT_INSTALMENT"
            END
        ) AS installment_payment_ratio,
        AVG(
            CASE
                WHEN "DAYS_ENTRY_PAYMENT" <= "DAYS_INSTALMENT"
                THEN 1.0
                ELSE 0.0
            END
        ) AS installment_ontime_ratio
    FROM installments_payments
    GROUP BY "SK_ID_CURR"
),

pos_cash_agg AS (

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
    GROUP BY "SK_ID_CURR"
),

credit_card_agg AS (

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
    GROUP BY "SK_ID_CURR"
),

bureau_monthly AS (

    SELECT
        "SK_ID_BUREAU",
        COUNT(*) AS bureau_balance_month_count,
        COUNT(*) FILTER (
            WHERE "STATUS" IN ('1', '2', '3', '4', '5')
        ) AS bureau_balance_dpd_month_count,
        COUNT(*) FILTER (
            WHERE "STATUS" = 'C'
        ) AS bureau_balance_closed_month_count,
        COUNT(*) FILTER (
            WHERE "STATUS" = '0'
        ) AS bureau_balance_current_month_count
    FROM bureau_balance
    GROUP BY "SK_ID_BUREAU"
),

bureau_balance_agg AS (

    SELECT
        b."SK_ID_CURR",
        COUNT(*) AS bureau_credit_count_with_balance_history,
        SUM(bb.bureau_balance_month_count)
            AS bureau_balance_total_months,
        SUM(bb.bureau_balance_dpd_month_count)
            AS bureau_balance_total_dpd_months,
        SUM(bb.bureau_balance_closed_month_count)
            AS bureau_balance_total_closed_months,
        SUM(bb.bureau_balance_current_month_count)
            AS bureau_balance_total_current_months,
        AVG(bb.bureau_balance_month_count)
            AS bureau_balance_avg_months_per_credit
    FROM bureau b
    INNER JOIN bureau_monthly bb
        ON b."SK_ID_BUREAU" = bb."SK_ID_BUREAU"
    GROUP BY b."SK_ID_CURR"
)

SELECT
    a.*,

    ba.bureau_loan_count,
    ba.bureau_avg_credit,
    ba.bureau_total_credit,
    ba.bureau_avg_debt,
    ba.bureau_total_debt,
    ba.bureau_avg_credit_limit,
    ba.bureau_total_credit_limit,
    ba.bureau_avg_annuity,
    ba.bureau_most_recent_days,
    ba.bureau_oldest_days,

    pa.previous_application_count,
    pa.previous_approved_count,
    pa.previous_refused_count,
    pa.previous_avg_credit,
    pa.previous_total_credit,
    pa.previous_avg_application,
    pa.previous_avg_credit_to_application_ratio,

    ia.installment_payment_count,
    ia.installment_avg_due,
    ia.installment_avg_paid,
    ia.installment_total_due,
    ia.installment_total_paid,
    ia.installment_payment_ratio,
    ia.installment_ontime_ratio,

    pc.pos_cash_record_count,
    pc.pos_cash_avg_months_balance,
    pc.pos_cash_oldest_month,
    pc.pos_cash_avg_installments,
    pc.pos_cash_avg_future_installments,
    pc.pos_cash_avg_dpd,
    pc.pos_cash_max_dpd,
    pc.pos_cash_avg_dpd_def,
    pc.pos_cash_max_dpd_def,
    pc.pos_cash_active_count,
    pc.pos_cash_completed_count,

    cc.credit_card_record_count,
    cc.credit_card_avg_months_balance,
    cc.credit_card_oldest_month,
    cc.credit_card_avg_balance,
    cc.credit_card_max_balance,
    cc.credit_card_avg_limit,
    cc.credit_card_max_limit,
    cc.credit_card_avg_drawings,
    cc.credit_card_total_drawings,
    cc.credit_card_avg_payment,
    cc.credit_card_total_payment,
    cc.credit_card_avg_receivable,
    cc.credit_card_avg_dpd,
    cc.credit_card_max_dpd,
    cc.credit_card_avg_dpd_def,
    cc.credit_card_max_dpd_def,

    bba.bureau_credit_count_with_balance_history,
    bba.bureau_balance_total_months,
    bba.bureau_balance_total_dpd_months,
    bba.bureau_balance_total_closed_months,
    bba.bureau_balance_total_current_months,
    bba.bureau_balance_avg_months_per_credit

FROM application_train a

LEFT JOIN bureau_agg ba
    ON a."SK_ID_CURR" = ba."SK_ID_CURR"

LEFT JOIN previous_application_agg pa
    ON a."SK_ID_CURR" = pa."SK_ID_CURR"

LEFT JOIN installments_agg ia
    ON a."SK_ID_CURR" = ia."SK_ID_CURR"

LEFT JOIN pos_cash_agg pc
    ON a."SK_ID_CURR" = pc."SK_ID_CURR"

LEFT JOIN credit_card_agg cc
    ON a."SK_ID_CURR" = cc."SK_ID_CURR"

LEFT JOIN bureau_balance_agg bba
    ON a."SK_ID_CURR" = bba."SK_ID_CURR";