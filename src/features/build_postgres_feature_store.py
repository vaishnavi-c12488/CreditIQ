"""
CreditIQ — PostgreSQL Feature Store Builder

Builds the applicant-level PostgreSQL feature store using the
same TRAIN-only WOE/IV transformation logic used by the
existing CreditIQ model-development pipeline.

Important:
- WOE/IV is learned from TRAIN only.
- Validation/test are never used to learn mappings.
- Full applicant population is transformed using TRAIN mappings.
- The final Feature Store contains exactly the 80 features
  expected by the finalized Monotonic XGBoost model.
"""

from pathlib import Path
import json
import joblib
import pandas as pd
from sqlalchemy import text

from src.features.woe_encoder import calculate_woe_iv
from src.data.db_loader import create_database_engine


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train.parquet"
)

ENGINEERED_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "applicant_features_engineered.parquet"
)

FINAL_TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train_final.parquet"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "xgboost_monotonic.joblib"
)

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "models"
    / "feature_store"
)

WOE_ARTIFACT_FILE = (
    ARTIFACT_DIR
    / "woe_transform_contract.joblib"
)

METADATA_FILE = (
    ARTIFACT_DIR
    / "feature_store_metadata.json"
)


# ============================================================
# CONSTANTS
# ============================================================

ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"

FEATURE_SNAPSHOT_VERSION = "v1.0"

NEUTRAL_WOE = 0.0


# ============================================================
# HELPERS
# ============================================================

def build_train_mapping(train: pd.DataFrame, feature: str):
    """
    Recreate the exact TRAIN-only WOE mapping used by
    the existing WOE pipeline.
    """

    result = calculate_woe_iv(
        train,
        feature,
        target=TARGET_COLUMN,
    )

    if result is None:
        return None

    grouped, iv, bin_edges, binning_type = result

    mapping = dict(
        zip(
            grouped["bin"].astype(str),
            grouped["woe"],
        )
    )

    contract = {
        "feature": feature,
        "iv": float(iv),
        "binning_type": binning_type,
        "mapping": mapping,
        "bin_edges": bin_edges,
    }

    return contract


def transform_feature(
    df: pd.DataFrame,
    contract: dict,
) -> pd.Series:
    """
    Apply an already learned TRAIN WOE contract to a dataframe.
    """

    feature = contract["feature"]
    mapping = contract["mapping"]
    binning_type = contract["binning_type"]

    # --------------------------------------------------------
    # Numeric feature
    # --------------------------------------------------------

    if binning_type == "continuous":

        bin_edges = contract["bin_edges"]

        bins = pd.cut(
            df[feature],
            bins=bin_edges,
            include_lowest=True,
        )

        keys = bins.astype(str)

        transformed = keys.map(mapping)

        # Learned missing-value WOE
        missing_woe = mapping.get("__MISSING__")

        if missing_woe is not None:

            transformed.loc[
                df[feature].isna()
            ] = missing_woe

        # Same fallback principle as existing pipeline
        numeric_woe = pd.Series(
            [
                value
                for key, value in mapping.items()
                if key != "__MISSING__"
            ],
            dtype="float64",
        )

        if not numeric_woe.empty:

            out_of_range_woe = float(
                numeric_woe.mean()
            )

            transformed = transformed.fillna(
                out_of_range_woe
            )

        else:

            transformed = transformed.fillna(
                NEUTRAL_WOE
            )

        return transformed.astype(float)

    # --------------------------------------------------------
    # Categorical feature
    # --------------------------------------------------------

    keys = (
        df[feature]
        .fillna("__MISSING__")
        .astype(str)
    )

    transformed = keys.map(mapping)

    # Same neutral fallback as existing WOE pipeline
    transformed = transformed.fillna(
        NEUTRAL_WOE
    )

    return transformed.astype(float)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("CreditIQ — PostgreSQL FEATURE STORE BUILDER")
    print("=" * 60)

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load model feature contract
    # --------------------------------------------------------

    print("\nLoading finalized model...")

    model = joblib.load(
        MODEL_FILE
    )

    model_features = list(
        model.feature_names_in_
    )

    print(
        f"Model features: {len(model_features)}"
    )

    if len(model_features) != 80:

        raise ValueError(
            "Expected exactly 80 model features."
        )

    # --------------------------------------------------------
    # Load training data
    # --------------------------------------------------------

    print("\nLoading frozen training data...")

    train = pd.read_parquet(
        TRAIN_FILE
    )

    print(
        f"Training rows: {len(train):,}"
    )

    # --------------------------------------------------------
    # Load complete engineered population
    # --------------------------------------------------------

    print(
        "\nLoading complete engineered applicant population..."
    )

    population = pd.read_parquet(
        ENGINEERED_FILE
    )

    print(
        f"Population rows: {len(population):,}"
    )

    # --------------------------------------------------------
    # Validate applicant population
    # --------------------------------------------------------

    if ID_COLUMN not in population.columns:

        raise ValueError(
            f"{ID_COLUMN} missing from engineered dataset."
        )

    if population[ID_COLUMN].isna().any():

        raise ValueError(
            "Applicant IDs contain missing values."
        )

    if population[ID_COLUMN].duplicated().any():

        raise ValueError(
            "Duplicate applicant IDs found."
        )

    print(
        f"Unique applicants: "
        f"{population[ID_COLUMN].nunique():,}"
    )

    # --------------------------------------------------------
    # Validate model features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in model_features
        if feature not in population.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing model features:\n"
            + "\n".join(missing_features)
        )

    print(
        "PASS: All 80 model features exist."
    )

    # --------------------------------------------------------
    # Learn TRAIN-only WOE contracts
    # --------------------------------------------------------

    print(
        "\nLearning TRAIN-only WOE mappings..."
    )

    contracts = {}

    for index, feature in enumerate(
        model_features,
        start=1,
    ):

        print(
            f"[{index:02d}/80] {feature}"
        )

        contract = build_train_mapping(
            train,
            feature,
        )

        if contract is None:

            raise ValueError(
                f"Unable to build WOE mapping "
                f"for feature: {feature}"
            )

        contracts[feature] = contract

    print(
        "\nPASS: TRAIN-only WOE contracts created."
    )

    # --------------------------------------------------------
    # Transform complete population
    # --------------------------------------------------------

    print(
        "\nTransforming complete applicant population..."
    )

    feature_store = population[
        [ID_COLUMN]
    ].copy()

    for index, feature in enumerate(
        model_features,
        start=1,
    ):

        print(
            f"Transforming [{index:02d}/80] {feature}"
        )

        feature_store[feature] = transform_feature(
            population,
            contracts[feature],
        )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    if TARGET_COLUMN in population.columns:

        feature_store[TARGET_COLUMN] = (
            population[TARGET_COLUMN]
            .astype("Int64")
        )

    else:

        feature_store[TARGET_COLUMN] = pd.NA

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    expected_columns = (
        [ID_COLUMN]
        + model_features
        + [TARGET_COLUMN]
    )

    feature_store = feature_store[
        expected_columns
    ]

    print(
        "\n============================================================"
    )

    print(
        "FEATURE STORE VALIDATION"
    )

    print(
        "============================================================"
    )

    print(
        f"Rows: {len(feature_store):,}"
    )

    print(
        f"Columns: {len(feature_store.columns):,}"
    )

    print(
        f"Model features: "
        f"{len(feature_store.columns) - 2:,}"
    )

    if len(feature_store) != len(population):

        raise ValueError(
            "Feature Store row count does not match population."
        )

    if feature_store[ID_COLUMN].duplicated().any():

        raise ValueError(
            "Feature Store contains duplicate applicants."
        )

    model_matrix = feature_store[
        model_features
    ]

    if not all(
        pd.api.types.is_numeric_dtype(
            model_matrix[column]
        )
        for column in model_features
    ):

        raise ValueError(
            "Not all model features are numeric."
        )

    if model_matrix.isna().any().any():

        missing_counts = (
            model_matrix.isna()
            .sum()
        )

        raise ValueError(
            "NaN values remain in model features:\n"
            + str(
                missing_counts[
                    missing_counts > 0
                ]
            )
        )

    print(
        "PASS: Row count validated."
    )

    print(
        "PASS: Applicant IDs are unique."
    )

    print(
        "PASS: All 80 model features are numeric."
    )

    print(
        "PASS: No NaN values remain."
    )

    # --------------------------------------------------------
    # Save WOE contract
    # --------------------------------------------------------

    print(
        "\nSaving WOE transformation contract..."
    )

    joblib.dump(
        {
            "version": FEATURE_SNAPSHOT_VERSION,
            "features": model_features,
            "contracts": contracts,
        },
        WOE_ARTIFACT_FILE,
    )

    print(
        f"Saved:\n{WOE_ARTIFACT_FILE}"
    )

    # --------------------------------------------------------
    # Save temporary local verification copy
    # --------------------------------------------------------

    local_feature_store = (
        PROJECT_ROOT
        / "data"
        / "interim"
        / "applicant_features_model_ready.parquet"
    )

    feature_store.to_parquet(
        local_feature_store,
        index=False,
    )

    print(
        f"\nSaved local model-ready Feature Store:\n"
        f"{local_feature_store}"
    )

    # --------------------------------------------------------
    # PostgreSQL
    # --------------------------------------------------------

    print(
        "\nConnecting to PostgreSQL..."
    )

    engine = create_database_engine()

    # --------------------------------------------------------
    # Recreate application-owned feature table
    # --------------------------------------------------------

    create_sql = """
    DROP TABLE IF EXISTS applicant_features;

    CREATE TABLE applicant_features (
        sk_id_curr INTEGER PRIMARY KEY,
        feature_snapshot_version VARCHAR(20) NOT NULL,
        target INTEGER,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """

    with engine.begin() as connection:

        for statement in create_sql.split(";"):

            statement = statement.strip()

            if statement:

                connection.execute(
                    text(statement)
                )

    # --------------------------------------------------------
    # Add the exact 80 feature columns
    # --------------------------------------------------------

    print(
        "\nCreating 80 model feature columns..."
    )

    with engine.begin() as connection:

        for feature in model_features:

            connection.execute(
                text(
                    f'''
                    ALTER TABLE applicant_features
                    ADD COLUMN "{feature}" DOUBLE PRECISION
                    '''
                )
            )

    # --------------------------------------------------------
    # Insert feature store
    # --------------------------------------------------------

    print(
        "\nWriting Feature Store to PostgreSQL..."
    )

    db_df = feature_store.copy()

    db_df = db_df.rename(
        columns={
            ID_COLUMN: "sk_id_curr",
            TARGET_COLUMN: "target",
        }
    )

    db_df.insert(
        1,
        "feature_snapshot_version",
        FEATURE_SNAPSHOT_VERSION,
    )

    db_df.to_sql(
        "applicant_features",
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=5000,
    )

    # --------------------------------------------------------
    # Database validation
    # --------------------------------------------------------

    with engine.connect() as connection:

        row_count = connection.execute(
            text(
                "SELECT COUNT(*) "
                "FROM applicant_features"
            )
        ).scalar()

        distinct_count = connection.execute(
            text(
                "SELECT COUNT(DISTINCT sk_id_curr) "
                "FROM applicant_features"
            )
        ).scalar()

    print(
        "\n============================================================"
    )

    print(
        "POSTGRESQL FEATURE STORE VALIDATION"
    )

    print(
        "============================================================"
    )

    print(
        f"Database rows: {row_count:,}"
    )

    print(
        f"Unique applicants: {distinct_count:,}"
    )

    if row_count != len(feature_store):

        raise ValueError(
            "PostgreSQL row count does not match local Feature Store."
        )

    if distinct_count != len(feature_store):

        raise ValueError(
            "PostgreSQL applicant IDs are not unique."
        )

    print(
        "PASS: PostgreSQL row count matches."
    )

    print(
        "PASS: PostgreSQL applicant IDs are unique."
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "feature_snapshot_version": FEATURE_SNAPSHOT_VERSION,
        "rows": int(len(feature_store)),
        "model_features": int(len(model_features)),
        "model_feature_names": model_features,
        "source_population": str(
            ENGINEERED_FILE.relative_to(
                PROJECT_ROOT
            )
        ),
        "woe_artifact": str(
            WOE_ARTIFACT_FILE.relative_to(
                PROJECT_ROOT
            )
        ),
        "model": str(
            MODEL_FILE.relative_to(
                PROJECT_ROOT
            )
        ),
        "woe_training_source": str(
            TRAIN_FILE.relative_to(
                PROJECT_ROOT
            )
        ),
    }

    METADATA_FILE.write_text(
        json.dumps(
            metadata,
            indent=4,
        )
    )

    print(
        f"\nMetadata saved:\n{METADATA_FILE}"
    )

    engine.dispose()

    print(
        "\n============================================================"
    )

    print(
        "CreditIQ PostgreSQL Feature Store "
        "completed successfully."
    )

    print(
        "============================================================"
    )


if __name__ == "__main__":
    main()