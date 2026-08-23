WITH bureau_monthly AS (

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
)

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

GROUP BY b."SK_ID_CURR";