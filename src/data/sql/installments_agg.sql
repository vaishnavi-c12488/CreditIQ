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

GROUP BY "SK_ID_CURR";