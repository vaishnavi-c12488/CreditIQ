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

GROUP BY "SK_ID_CURR";