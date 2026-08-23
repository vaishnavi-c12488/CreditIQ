from pathlib import Path
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env", override=True)

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "creditiq")

PARQUET_FILE = Path(
    "/app/applicant_features_model_ready.parquet"
)


def main():

    print("Loading model-ready Parquet...")

    df = pd.read_parquet(PARQUET_FILE)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # Rename ID to the exact API/database column name
    df = df.rename(
        columns={
            "SK_ID_CURR": "sk_id_curr"
        }
    )

    # TARGET is training metadata, not a model feature.
    # Keep it in the feature store because it is useful for
    # auditing/training lineage.
    df["feature_snapshot_version"] = "model_ready_v1"

    # PostgreSQL connection
    engine = create_engine(
        f"postgresql+psycopg2://"
        f"{DB_USER}:{DB_PASSWORD}@"
        f"{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    print("Connected to PostgreSQL.")

    # Create application tables
    schema_file = Path("/app/schema.sql")

    with engine.begin() as connection:
        connection.execute(
            text(schema_file.read_text(encoding="utf-8"))
        )

    print("Application tables created.")

    # Replace the placeholder applicant_features table
    # with the complete 82-column model-ready feature store.
    with engine.begin() as connection:
        connection.execute(
            text(
                "DROP TABLE IF EXISTS applicant_features "
                "CASCADE"
            )
        )

    print("Old applicant_features table removed.")

    # Load complete feature store
    print("Loading applicant_features...")

    df.to_sql(
        "applicant_features",
        engine,
        if_exists="replace",
        index=False,
        chunksize=5000,
        method="multi",
    )

    # Add primary key
    with engine.begin() as connection:

        connection.execute(
            text(
                """
                ALTER TABLE applicant_features
                ADD PRIMARY KEY (sk_id_curr)
                """
            )
        )

    print("Primary key created.")

    # Final validation
    with engine.connect() as connection:

        count = connection.execute(
            text(
                "SELECT COUNT(*) FROM applicant_features"
            )
        ).scalar()

        applicant = connection.execute(
            text(
                """
                SELECT sk_id_curr
                FROM applicant_features
                WHERE sk_id_curr = 447009
                """
            )
        ).first()

    print()
    print("=" * 60)
    print("Docker database seeding completed")
    print("=" * 60)
    print(f"Applicants loaded: {count:,}")
    print(f"Applicant 447009 exists: {applicant is not None}")


if __name__ == "__main__":
    main()