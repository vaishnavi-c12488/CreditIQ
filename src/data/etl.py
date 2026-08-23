from pathlib import Path
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SQL_FILE = (
    PROJECT_ROOT
    / "src"
    / "data"
    / "sql"
    / "final_applicant_features.sql"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "interim"
OUTPUT_FILE = OUTPUT_DIR / "applicant_features_raw.parquet"


# ============================================================
# Load environment variables
# ============================================================

load_dotenv(
    PROJECT_ROOT / ".env",
    override=True
)

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "creditiq")


# ============================================================
# Database connection
# ============================================================

DATABASE_URL = URL.create(
    drivername="postgresql+psycopg2",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=int(DB_PORT),
    database=DB_NAME
)

engine = create_engine(DATABASE_URL)


# ============================================================
# ETL
# ============================================================

def run_etl():

    print("Starting CreditIQ ETL...")

    if not SQL_FILE.exists():
        raise FileNotFoundError(
            f"SQL file not found: {SQL_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Reading final ETL SQL...")

    sql = SQL_FILE.read_text(
        encoding="utf-8"
    ).strip()

    print("Executing ETL query...")

    df = pd.read_sql_query(
        sql,
        engine
    )

    print(
        f"ETL produced {len(df):,} rows "
        f"and {len(df.columns):,} columns."
    )


    # ========================================================
    # Validation
    # ========================================================

    expected_rows = 307_511

    if len(df) != expected_rows:
        raise ValueError(
            f"Unexpected row count: {len(df):,}. "
            f"Expected {expected_rows:,}."
        )

    if df["SK_ID_CURR"].nunique() != expected_rows:
        raise ValueError(
            "SK_ID_CURR is not unique in the ETL output."
        )

    print("Row-count validation passed.")
    print("SK_ID_CURR uniqueness validation passed.")


    # ========================================================
    # Save Parquet
    # ========================================================

    df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"Saved ETL output to:\n{OUTPUT_FILE}"
    )

    print("\nCreditIQ ETL completed successfully.")


if __name__ == "__main__":
    run_etl()