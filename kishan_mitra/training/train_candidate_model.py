
import sys
import json
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from kishan_mitra.config import DATASET_PATH, TARGET_COLUMN
from training.model_config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TRAIN_END_YEAR,
    RANDOM_STATE,
)

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

CANDIDATE_MODEL_PATH = (
    PROJECT_ROOT / "models" / "crop_yield_extratrees_candidate.joblib"
)
CANDIDATE_METADATA_PATH = (
    PROJECT_ROOT / "models" / "crop_yield_extratrees_candidate_metadata.json"
)


def build_pipeline():
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ])

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer([
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
    ])

    model = ExtraTreesRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH, low_memory=False)

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN], errors="coerce"
    )

    df = df.dropna(subset=["Year", TARGET_COLUMN, "Crop"])
    df = df[df[TARGET_COLUMN] > 0].copy()

    # Keep the same training cutoff used in the earlier model comparison.
    train = df[df["Year"] <= TRAIN_END_YEAR].copy()

    if train.empty:
        raise ValueError("Training dataset is empty.")

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]

    print(f"Training rows: {len(train):,}")
    print(
        f"Training years: {int(train['Year'].min())}-"
        f"{int(train['Year'].max())}"
    )
    print("Model: Extra Trees Regressor")
    print("Parameters: n_estimators=300, min_samples_leaf=2")
    print("Fitting pipeline...")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Verify the fitted pipeline produces finite predictions before saving.
    sample_predictions = pipeline.predict(X_train.head(10))

    if len(sample_predictions) != min(10, len(X_train)):
        raise RuntimeError("Unexpected number of sample predictions.")

    if not np.isfinite(sample_predictions).all():
        raise RuntimeError("Candidate model produced non-finite predictions.")

    PROJECT_ROOT.joinpath("models").mkdir(
        parents=True, exist_ok=True
    )

    # Save to a NEW path. The existing Random Forest artifact is untouched.
    joblib.dump(pipeline, CANDIDATE_MODEL_PATH)

    metadata = {
        "model_name": "Extra Trees Regressor",
        "artifact_type": "candidate",
        "status": "trained_not_promoted",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_column": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "training_rows": int(len(train)),
        "training_start_year": int(train["Year"].min()),
        "training_end_year": int(train["Year"].max()),
        "parameters": {
            "n_estimators": 300,
            "min_samples_leaf": 2,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        },
        "preprocessing": {
            "categorical_imputation": "most_frequent",
            "categorical_encoding": "OneHotEncoder(handle_unknown='ignore')",
            "numeric_imputation": "median",
            "numeric_scaling": "StandardScaler",
        },
        "test_metrics_from_prior_evaluation": {
            "MAE": 441.8932,
            "RMSE": 810.4445,
            "R2": 0.6825,
        },
        "note": (
            "Previously reported test metrics are from the earlier "
            "evaluation run. They are recorded for reference and are "
            "not recalculated or independently verified by this script."
        ),
    }

    with open(CANDIDATE_METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\nCandidate model saved successfully.")
    print(f"Model artifact: {CANDIDATE_MODEL_PATH}")
    print(f"Metadata:       {CANDIDATE_METADATA_PATH}")
    print("Sample prediction check: PASSED")
    print("\nThe existing Random Forest artifact was not modified.")


if __name__ == "__main__":
    main()
