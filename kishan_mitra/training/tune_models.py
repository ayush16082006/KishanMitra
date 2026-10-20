
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lightgbm import LGBMRegressor

# ---------------------------------------------------------
# Project paths and imports
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

# Number of randomly sampled hyperparameter combinations.
EXTRA_TREES_ITERATIONS = 4
LIGHTGBM_ITERATIONS = 5

# Chronological CV folds, all inside the training period.
# Format: (last training year, validation start, validation end)
CV_WINDOWS = [
    (2000, 2001, 2004),
    (2004, 2005, 2008),
    (2008, 2009, 2012),
]


def build_preprocessor():
    """Create the same feature preprocessing used by the baseline."""

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

    return ColumnTransformer([
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES,
        ),
        (
            "numeric",
            numeric_pipeline,
            NUMERIC_FEATURES,
        ),
    ])


def build_pipeline(model):
    """Keep preprocessing inside the pipeline to prevent CV leakage."""

    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", model),
    ])


def calculate_metrics(actual, predicted):
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def build_chronological_cv_splits(train):
    """Create year-based CV folds instead of randomly splitting rows."""

    years = train["Year"].to_numpy()
    splits = []

    for last_train_year, val_start, val_end in CV_WINDOWS:
        train_indices = np.flatnonzero(years <= last_train_year)

        validation_indices = np.flatnonzero(
            (years >= val_start) & (years <= val_end)
        )

        if len(train_indices) == 0 or len(validation_indices) == 0:
            raise ValueError(
                f"Empty CV fold: train through {last_train_year}, "
                f"validate {val_start}-{val_end}"
            )

        splits.append((train_indices, validation_indices))

        print(
            f"CV fold: train through {last_train_year} "
            f"({len(train_indices):,} rows), "
            f"validate {val_start}-{val_end} "
            f"({len(validation_indices):,} rows)"
        )

    return splits


def tune_model(
    name,
    estimator,
    parameter_distributions,
    iterations,
    X_train,
    y_train,
    cv_splits,
):
    """Tune one model using training-only chronological CV."""

    print("\n" + "=" * 65)
    print(f"TUNING: {name}")
    print("=" * 65)

    search = RandomizedSearchCV(
        estimator=build_pipeline(estimator),
        param_distributions=parameter_distributions,
        n_iter=iterations,
        scoring="neg_mean_absolute_error",
        cv=cv_splits,
        refit=True,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=2,
        return_train_score=False,
        error_score="raise",
    )

    start = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start

    cv_results = pd.DataFrame(search.cv_results_)

    print(f"\nBest cross-validation MAE: {-search.best_score_:.4f}")
    print(f"Total tuning time: {elapsed:.1f} seconds")
    print("Best parameters:")

    for parameter, value in search.best_params_.items():
        print(f"  {parameter}: {value}")

    trial_results = pd.DataFrame({
        "Model": name,
        "CV_MAE": -cv_results["mean_test_score"],
        "CV_MAE_Std": cv_results["std_test_score"],
        "Mean_Fit_Seconds": cv_results["mean_fit_time"],
        "Parameters": cv_results["params"].map(str),
    }).sort_values("CV_MAE")

    return search.best_estimator_, search.best_params_, trial_results


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH, low_memory=False)

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN], errors="coerce"
    )

    df = df.dropna(subset=["Year", TARGET_COLUMN, "Crop"])
    df = df[df[TARGET_COLUMN] > 0].copy()

    # Strict chronological split.
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

    print("\n=== CHRONOLOGICAL CROSS-VALIDATION FOLDS ===")
    cv_splits = build_chronological_cv_splits(train)

    # -----------------------------------------------------
    # Candidate 1: Extra Trees
    # -----------------------------------------------------

    extra_trees_params = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 30],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": [0.8, 1.0],
    }

    # -----------------------------------------------------
    # Candidate 2: LightGBM
    # -----------------------------------------------------

    lightgbm_params = {
        "model__n_estimators": [200, 400, 600],
        "model__learning_rate": [0.03, 0.05, 0.08],
        "model__num_leaves": [15, 31, 63],
        "model__min_child_samples": [10, 20, 40],
        "model__reg_lambda": [0.0, 1.0, 5.0],
    }

    extra_model, extra_params, extra_trials = tune_model(
        name="Extra Trees",
        estimator=ExtraTreesRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        parameter_distributions=extra_trees_params,
        iterations=EXTRA_TREES_ITERATIONS,
        X_train=X_train,
        y_train=y_train,
        cv_splits=cv_splits,
    )

    light_model, light_params, light_trials = tune_model(
        name="LightGBM",
        estimator=LGBMRegressor(
            objective="regression",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        ),
        parameter_distributions=lightgbm_params,
        iterations=LIGHTGBM_ITERATIONS,
        X_train=X_train,
        y_train=y_train,
        cv_splits=cv_splits,
    )

    # -----------------------------------------------------
    # Validation evaluation: 2013-2014
    # -----------------------------------------------------

    models = {
        "Tuned Extra Trees": extra_model,
        "Tuned LightGBM": light_model,
    }

    validation_results = []
    crop_results = []

    print("\n" + "=" * 65)
    print("FINAL VALIDATION COMPARISON")
    print("=" * 65)

    for name, model in models.items():
        predictions = model.predict(X_validation)
        metrics = calculate_metrics(y_validation, predictions)

        validation_results.append({
            "Model": name,
            **metrics,
        })

        print(f"\n{name}")
        print(f"MAE:  {metrics['MAE']:.4f}")
        print(f"RMSE: {metrics['RMSE']:.4f}")
        print(f"R2:   {metrics['R2']:.4f}")

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

    validation_results_df = pd.DataFrame(
        validation_results
    ).sort_values("MAE")

    crop_results_df = pd.DataFrame(crop_results)

    # -----------------------------------------------------
    # Save reports; do not save or replace model artifacts.
    # -----------------------------------------------------

    output_dir = PROJECT_ROOT / "reports" / "model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    cv_results_df = pd.concat(
        [extra_trials, light_trials],
        ignore_index=True,
    ).sort_values("CV_MAE")

    cv_results_path = output_dir / "tuning_cv_results.csv"
    validation_path = output_dir / "tuned_validation_comparison.csv"
    crop_path = output_dir / "tuned_validation_crop_metrics.csv"
    params_path = output_dir / "tuning_best_params.json"

    cv_results_df.to_csv(cv_results_path, index=False)
    validation_results_df.to_csv(validation_path, index=False)
    crop_results_df.to_csv(crop_path, index=False)

    with open(params_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "Extra Trees": extra_params,
                "LightGBM": light_params,
                "cross_validation_windows": [
                    {
                        "train_through": last_train,
                        "validation_start": val_start,
                        "validation_end": val_end,
                    }
                    for last_train, val_start, val_end in CV_WINDOWS
                ],
                "selection_metric": "MAE",
                "random_state": RANDOM_STATE,
            },
            file,
            indent=4,
            default=str,
        )

    print("\n=== TUNED VALIDATION RANKING ===")
    print(validation_results_df.round(4).to_string(index=False))

    print("\n=== VALIDATION MAE BY CROP ===")
    print(
        crop_results_df.pivot(
            index="Crop",
            columns="Model",
            values="MAE",
        ).round(2).to_string()
    )

    print("\n=== REPORTS SAVED ===")
    print(cv_results_path)
    print(validation_path)
    print(crop_path)
    print(params_path)

    print(
        "\nNo model artifact was overwritten. "
        "The 2015-2017 test period was not used."
    )


if __name__ == "__main__":
    main()
