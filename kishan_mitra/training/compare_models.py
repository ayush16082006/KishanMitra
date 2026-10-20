
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from kishan_mitra.config import DATASET_PATH, TARGET_COLUMN
from training.model_config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TRAIN_END_YEAR,
    VALIDATION_START_YEAR,
    VALIDATION_END_YEAR,
    RANDOM_STATE,
)


FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def build_preprocessor(dense=False):
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=not dense,
            ),
        ),
    ])

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    return ColumnTransformer(
        [
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ],
        sparse_threshold=0 if dense else 1.0,
    )


def build_pipeline(model, dense=False):
    return Pipeline([
        ("preprocessor", build_preprocessor(dense=dense)),
        ("model", model),
    ])


def calculate_metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
    }


def prepare_catboost_frame(frame):
    result = frame[FEATURE_COLUMNS].copy()

    for column in CATEGORICAL_FEATURES:
        result[column] = (
            result[column].fillna("__MISSING__").astype(str)
        )

    for column in NUMERIC_FEATURES:
        result[column] = pd.to_numeric(
            result[column], errors="coerce"
        )

    return result


def main():
    print("Loading dataset...")

    df = pd.read_csv(DATASET_PATH, low_memory=False)

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN], errors="coerce"
    )

    df = df.dropna(subset=["Year", TARGET_COLUMN, "Crop"])
    df = df[df[TARGET_COLUMN] > 0].copy()

    train = df[df["Year"] <= TRAIN_END_YEAR].copy()

    validation = df[
        df["Year"].between(
            VALIDATION_START_YEAR,
            VALIDATION_END_YEAR,
        )
    ].copy()

    if train.empty or validation.empty:
        raise ValueError("Training or validation split is empty.")

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]

    X_validation = validation[FEATURE_COLUMNS]
    y_validation = validation[TARGET_COLUMN]

    # CatBoost receives categorical columns directly.
    X_train_cat = prepare_catboost_frame(train)
    X_validation_cat = prepare_catboost_frame(validation)

    candidates = {
        "Ridge": build_pipeline(
            Ridge(alpha=10.0)
        ),

        "Random Forest": build_pipeline(
            RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),

        "Extra Trees": build_pipeline(
            ExtraTreesRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),

        "Gradient Boosting": build_pipeline(
            GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=2,
                loss="huber",
                random_state=RANDOM_STATE,
            )
        ),

        "HistGradientBoosting": build_pipeline(
            HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.08,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                random_state=RANDOM_STATE,
            ),
            dense=True,
        ),

        "XGBoost": build_pipeline(
            XGBRegressor(
                objective="reg:squarederror",
                n_estimators=400,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                tree_method="hist",
            )
        ),

        "LightGBM": build_pipeline(
            LGBMRegressor(
                objective="regression",
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=-1,
                min_child_samples=20,
                verbosity=-1,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),

        "Random Forest (log target)": TransformedTargetRegressor(
            regressor=build_pipeline(
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
            ),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
    }

    results = []
    crop_results = []

    print("\n=== DATA SPLIT ===")
    print(f"Training rows: {len(train):,}")
    print(
        f"Training years: {int(train['Year'].min())}-"
        f"{int(train['Year'].max())}"
    )
    print(f"Validation rows: {len(validation):,}")
    print(
        f"Validation years: {VALIDATION_START_YEAR}-"
        f"{VALIDATION_END_YEAR}"
    )

    for name, model in candidates.items():
        print(f"\n{'=' * 55}")
        print(f"Training: {name}")
        start = time.time()

        model.fit(X_train, y_train)

        predictions = model.predict(X_validation)
        elapsed = time.time() - start

        metrics = calculate_metrics(y_validation, predictions)

        results.append({
            "Model": name,
            **metrics,
            "Training_Seconds": round(elapsed, 2),
            "Status": "success",
        })

        print(f"MAE:  {metrics['MAE']:.4f}")
        print(f"RMSE: {metrics['RMSE']:.4f}")
        print(f"R2:   {metrics['R2']:.4f}")
        print(f"Elapsed seconds: {elapsed:.1f}")

        for crop in sorted(validation["Crop"].unique()):
            mask = validation["Crop"].eq(crop).to_numpy()

            crop_metrics = calculate_metrics(
                y_validation.to_numpy()[mask],
                predictions[mask],
            )

            crop_results.append({
                "Model": name,
                "Crop": crop,
                "Rows": int(mask.sum()),
                **crop_metrics,
            })

    # CatBoost uses native categorical handling.
    print(f"\n{'=' * 55}")
    print("Training: CatBoost")
    start = time.time()

    catboost_model = CatBoostRegressor(
        iterations=600,
        learning_rate=0.05,
        depth=8,
        loss_function="RMSE",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )

    catboost_model.fit(
        X_train_cat,
        y_train,
        cat_features=CATEGORICAL_FEATURES,
    )

    catboost_predictions = catboost_model.predict(X_validation_cat)
    elapsed = time.time() - start

    metrics = calculate_metrics(y_validation, catboost_predictions)

    results.append({
        "Model": "CatBoost",
        **metrics,
        "Training_Seconds": round(elapsed, 2),
        "Status": "success",
    })

    print(f"MAE:  {metrics['MAE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"R2:   {metrics['R2']:.4f}")
    print(f"Elapsed seconds: {elapsed:.1f}")

    for crop in sorted(validation["Crop"].unique()):
        mask = validation["Crop"].eq(crop).to_numpy()

        crop_metrics = calculate_metrics(
            y_validation.to_numpy()[mask],
            catboost_predictions[mask],
        )

        crop_results.append({
            "Model": "CatBoost",
            "Crop": crop,
            "Rows": int(mask.sum()),
            **crop_metrics,
        })

    results_df = pd.DataFrame(results).sort_values("MAE")
    crop_df = pd.DataFrame(crop_results)

    output_dir = PROJECT_ROOT / "reports" / "model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "validation_model_comparison.csv"
    crop_path = output_dir / "validation_crop_comparison.csv"

    results_df.to_csv(results_path, index=False)
    crop_df.to_csv(crop_path, index=False)

    print("\n=== VALIDATION MODEL RANKING ===")
    print(results_df.round(4).to_string(index=False))

    print("\n=== VALIDATION MAE BY CROP ===")
    print(
        crop_df.pivot(
            index="Crop",
            columns="Model",
            values="MAE",
        ).round(2).to_string()
    )

    print("\n=== REPORTS SAVED ===")
    print(results_path)
    print(crop_path)

    print("\nLower MAE and RMSE are better; higher R2 is better.")
    print(
        "This script compares validation performance only. "
        "It does not overwrite or replace the current model."
    )


if __name__ == "__main__":
    main()
