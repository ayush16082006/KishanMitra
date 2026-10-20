
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from kishan_mitra.config import (
    DATASET_PATH,
    MODEL_ARTIFACT_PATH,
    MODEL_METADATA_PATH,
    TARGET_COLUMN,
    RANDOM_STATE,
)
from training.model_config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    FEATURE_COLUMNS,
    TRAIN_END_YEAR,
    VALIDATION_START_YEAR,
    VALIDATION_END_YEAR,
    TEST_START_YEAR,
    TEST_END_YEAR,
)


def load_dataset():
    df = pd.read_csv(DATASET_PATH, low_memory=False)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required dataset columns: {missing}")

    df = df[required_columns].copy()

    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

    # Keep rows with valid targets and years.
    df = df.dropna(subset=[TARGET_COLUMN, "Year"])
    df = df[df[TARGET_COLUMN] > 0]

    return df


def build_pipeline():
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def calculate_metrics(model, X, y):
    predictions = model.predict(X)

    return {
        "rows": int(len(y)),
        "MAE": float(mean_absolute_error(y, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y, predictions))),
        "R2": float(r2_score(y, predictions)),
    }


def main():
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="sklearn",
    )

    df = load_dataset()

    train = df[df["Year"] <= TRAIN_END_YEAR]
    validation = df[
        df["Year"].between(VALIDATION_START_YEAR, VALIDATION_END_YEAR)
    ]
    test = df[df["Year"].between(TEST_START_YEAR, TEST_END_YEAR)]

    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "At least one chronological split is empty. "
            "Check the year ranges in model_config.py."
        )

    print("\n=== CHRONOLOGICAL DATA SPLIT ===")
    print(f"Training rows (through {TRAIN_END_YEAR}): {len(train):,}")
    print(
        f"Validation rows ({VALIDATION_START_YEAR}-"
        f"{VALIDATION_END_YEAR}): {len(validation):,}"
    )
    print(
        f"Test rows ({TEST_START_YEAR}-{TEST_END_YEAR}): {len(test):,}"
    )

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]

    X_validation = validation[FEATURE_COLUMNS]
    y_validation = validation[TARGET_COLUMN]

    X_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMN]

    pipeline = build_pipeline()

    print("\nTraining Random Forest...")
    pipeline.fit(X_train, y_train)

    print("\n=== VALIDATION METRICS ===")
    validation_metrics = calculate_metrics(
        pipeline, X_validation, y_validation
    )
    for name, value in validation_metrics.items():
        print(f"{name}: {value:,.4f}" if isinstance(value, float) else f"{name}: {value}")

    # Final test is evaluated only after training and validation.
    print("\n=== FINAL TEST METRICS ===")
    test_metrics = calculate_metrics(pipeline, X_test, y_test)
    for name, value in test_metrics.items():
        print(f"{name}: {value:,.4f}" if isinstance(value, float) else f"{name}: {value}")

    MODEL_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, MODEL_ARTIFACT_PATH)

    metadata = {
        "target_column": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "training_year_end": TRAIN_END_YEAR,
        "validation_years": [
            VALIDATION_START_YEAR,
            VALIDATION_END_YEAR,
        ],
        "test_years": [TEST_START_YEAR, TEST_END_YEAR],
        "model_type": "RandomForestRegressor",
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }

    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\n=== MODEL SAVED ===")
    print(f"Pipeline: {MODEL_ARTIFACT_PATH}")
    print(f"Metadata: {MODEL_METADATA_PATH}")


if __name__ == "__main__":
    main()
