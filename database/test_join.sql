SELECT
    COUNT(*) AS joined_rows,
    COUNT(DISTINCT a."SK_ID_CURR") AS unique_applicants
FROM application_train AS a
INNER JOIN bureau AS b
    ON a."SK_ID_CURR" = b."SK_ID_CURR";