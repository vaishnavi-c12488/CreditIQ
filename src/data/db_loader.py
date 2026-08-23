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
RAW_DIR = PROJECT_ROOT / "data" / "raw"


# ============================================================
# Load environment variables
# ============================================================

load_dotenv(PROJECT_ROOT / ".env")


# ============================================================
# PostgreSQL connection
# ============================================================

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")


DATABASE_URL = URL.create(
    drivername="postgresql+psycopg2",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=int(DB_PORT),
    database=DB_NAME,
)


# ============================================================
# Raw CSV → PostgreSQL table mapping
# ============================================================

RAW_TABLES = {
    "application_train.csv": "application_train",
    "bureau.csv": "bureau",
    "bureau_balance.csv": "bureau_balance",
    "previous_application.csv": "previous_application",
    "POS_CASH_balance.csv": "pos_cash_balance",
    "installments_payments.csv": "installments_payments",
    "credit_card_balance.csv": "credit_card_balance",
}


def create_database_engine():
    """Create a SQLAlchemy engine for PostgreSQL."""
    return create_engine(DATABASE_URL)


def load_raw_tables():
    """Load all seven raw CSV files into PostgreSQL."""

    engine = create_database_engine()

    try:
        for csv_file, table_name in RAW_TABLES.items():

            file_path = RAW_DIR / csv_file

            if not file_path.exists():
                raise FileNotFoundError(
                    f"Required dataset not found: {file_path}"
                )

            print("\n" + "=" * 60)
            print(f"Loading: {csv_file}")
            print(f"Target table: {table_name}")
            print("=" * 60)

            df = pd.read_csv(file_path)

            print(f"Rows read: {len(df):,}")
            print(f"Columns: {len(df.columns):,}")

            df.to_sql(
                table_name,
                engine,
                if_exists="replace",
                index=False,
                chunksize=10_000,
                method="multi",
            )

            print(
                f"Successfully loaded {len(df):,} rows "
                f"into {table_name}"
            )

    finally:
        engine.dispose()

    print("\n" + "=" * 60)
    print("ALL RAW TABLES LOADED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    load_raw_tables()