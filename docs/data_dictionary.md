# CreditIQ — Data Dictionary

## 1. Purpose

This document describes the structure of the Home Credit dataset used in CreditIQ.

The purpose of this phase is to understand:
- what each table represents
- what one row represents (grain)
- the applicant identifier used to connect tables
- relationships between tables
- important business columns

No feature engineering or modeling is performed in this phase.

---

# 2. Entity Relationship Overview

The central table is:

`application_train`

The applicant identifier is:

`SK_ID_CURR`

The other six tables contain additional historical information related to applicants.

High-level relationship:

application_train
│
├── bureau
│   └── bureau_balance
│
├── previous_application
│
├── POS_CASH_balance
│
├── installments_payments
│
└── credit_card_balance

Most relationships are based on `SK_ID_CURR`.

---

# 3. application_train

## Grain

One row represents one current loan application / applicant.

## Key

`SK_ID_CURR`

## Target

`TARGET`

- `0` = applicant did not default
- `1` = applicant defaulted

## Important columns

| Column | Description |
|---|---|
| SK_ID_CURR | Unique applicant/application identifier |
| TARGET | Current loan repayment outcome |
| NAME_CONTRACT_TYPE | Type of credit product requested |
| CODE_GENDER | Applicant gender |
| FLAG_OWN_CAR | Whether applicant owns a car |
| FLAG_OWN_REALTY | Whether applicant owns real estate |
| CNT_CHILDREN | Number of children |
| AMT_INCOME_TOTAL | Applicant's total income |
| AMT_CREDIT | Credit amount requested |
| AMT_ANNUITY | Loan annuity/payment amount |
| AMT_GOODS_PRICE | Price of goods associated with the loan |
| NAME_INCOME_TYPE | Applicant's income/employment category |
| NAME_EDUCATION_TYPE | Applicant's education level |
| NAME_FAMILY_STATUS | Applicant's family status |
| NAME_HOUSING_TYPE | Applicant's housing situation |
| DAYS_BIRTH | Applicant age represented in days |
| DAYS_EMPLOYED | Employment duration represented in days |
| OCCUPATION_TYPE | Applicant occupation |

---

# 4. bureau

## Grain

One row represents one previous credit account reported by an external credit bureau for an applicant.

## Key relationship

`SK_ID_CURR`

## Important columns

| Column | Description |
|---|---|
| SK_ID_CURR | Applicant identifier |
| SK_ID_BUREAU | Unique bureau credit record |
| CREDIT_ACTIVE | Status of the bureau credit |
| CREDIT_CURRENCY | Currency of the credit |
| DAYS_CREDIT | Days since bureau credit was reported |
| CREDIT_DAY_OVERDUE | Number of days credit was overdue |
| AMT_CREDIT_SUM | Total credit amount |
| AMT_CREDIT_SUM_DEBT | Current debt amount |
| AMT_CREDIT_SUM_OVERDUE | Amount currently overdue |
| CNT_CREDIT_PROLONG | Number of times credit was prolonged |

---

# 5. bureau_balance

## Grain

One row represents one monthly balance record for a bureau credit account.

## Key relationship

`SK_ID_BUREAU`

## Important columns

| Column | Description |
|---|---|
| SK_ID_BUREAU | Bureau credit identifier |
| MONTHS_BALANCE | Month relative to the current application |
| STATUS | Monthly status of the bureau credit |

`bureau_balance` connects to `bureau` through `SK_ID_BUREAU`.

---

# 6. previous_application

## Grain

One row represents one previous loan application made by an applicant.

## Key relationship

`SK_ID_CURR`

## Important columns

| Column | Description |
|---|---|
| SK_ID_PREV | Unique previous application identifier |
| SK_ID_CURR | Current applicant identifier |
| NAME_CONTRACT_TYPE | Type of previous credit |
| AMT_CREDIT | Amount of previous credit |
| AMT_ANNUITY | Previous loan annuity |
| AMT_APPLICATION | Amount requested in previous application |
| AMT_DOWN_PAYMENT | Down payment amount |
| NAME_CONTRACT_STATUS | Status of previous application |
| DAYS_DECISION | Days since decision |
| NAME_PAYMENT_TYPE | Payment method |
| NAME_PRODUCT_TYPE | Type of financial product |

---

# 7. POS_CASH_balance

## Grain

One row represents one monthly POS/cash-loan balance record.

## Key relationship

`SK_ID_PREV`

## Important columns

| Column | Description |
|---|---|
| SK_ID_PREV | Previous application identifier |
| SK_ID_CURR | Applicant identifier |
| MONTHS_BALANCE | Month relative to current application |
| CNT_INSTALMENT | Number of installments |
| CNT_INSTALMENT_FUTURE | Future installments remaining |
| NAME_CONTRACT_STATUS | Contract status |
| SK_DPD | Days past due |
| SK_DPD_DEF | Days past due excluding certain deferrals |

---

# 8. installments_payments

## Grain

One row represents one installment/payment record associated with a previous application.

## Key relationship

`SK_ID_PREV`

## Important columns

| Column | Description |
|---|---|
| SK_ID_PREV | Previous application identifier |
| SK_ID_CURR | Applicant identifier |
| NUM_INSTALMENT_VERSION | Version of installment schedule |
| NUM_INSTALMENT_NUMBER | Installment sequence number |
| DAYS_INSTALMENT | Scheduled payment day |
| DAYS_ENTRY_PAYMENT | Actual payment day |
| AMT_INSTALMENT | Scheduled installment amount |
| AMT_PAYMENT | Actual payment amount |

---

# 9. credit_card_balance

## Grain

One row represents one monthly credit-card balance record for a previous credit-card application.

## Key relationship

`SK_ID_PREV`

## Important columns

| Column | Description |
|---|---|
| SK_ID_PREV | Previous application identifier |
| SK_ID_CURR | Applicant identifier |
| MONTHS_BALANCE | Month relative to current application |
| AMT_BALANCE | Credit-card balance |
| AMT_CREDIT_LIMIT_ACTUAL | Actual credit limit |
| AMT_DRAWINGS_CURRENT | Current drawings |
| AMT_PAYMENT_CURRENT | Current payment |
| CNT_DRAWINGS_CURRENT | Number of current drawings |
| CNT_INSTALMENT_MATURE_CURRE | Number of mature installments |
| SK_DPD | Days past due |

---

# 10. Relationship Summary

| Parent | Child | Relationship key |
|---|---|---|
| application_train | bureau | SK_ID_CURR |
| bureau | bureau_balance | SK_ID_BUREAU |
| application_train | previous_application | SK_ID_CURR |
| previous_application | POS_CASH_balance | SK_ID_PREV |
| previous_application | installments_payments | SK_ID_PREV |
| previous_application | credit_card_balance | SK_ID_PREV |

---

# 11. Important Data Understanding Notes

The dataset contains one-to-many relationships.

For example:

One applicant
→ many bureau records

One applicant
→ many previous applications

One previous application
→ many installment payments

One previous application
→ many POS/CASH balance records

One previous application
→ many credit-card balance records

Therefore, these tables cannot simply be joined directly to `application_train` without aggregation.

The ETL phase will aggregate these historical records to the applicant level before joining them to the main application table.
