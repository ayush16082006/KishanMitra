
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lightgbm import LGBMRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kishan_mitra.config import DATASET_PATH, TARGET_COLUMN, MODEL_ARTIFACT_PATH
from training.model_config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TRAIN_END_YEAR,
    RANDOM_STATE,
)

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TEST_START_YEAR = 2015
TEST_END_YEAR = 2017


def build_pipeline(model):
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer([
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def calculate_metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
        "Mean_Error": np.mean(predicted - actual),
    }


def crop_state_median_predictions(train, test):
    """Baseline medians are learned exclusively from training data."""

    overall_median = train[TARGET_COLUMN].median()

    crop_medians = train.groupby("Crop")[TARGET_COLUMN].median()

    crop_state_medians = train.groupby(
        ["Crop", "State Name"]
    )[TARGET_COLUMN].median()

    predictions = []

    for _, row in test.iterrows():
        key = (row["Crop"], row["State Name"])

        if key in crop_state_medians.index:
            prediction = crop_state_medians.loc[key]
        elif row["Crop"] in crop_medians.index:
            prediction = crop_medians.loc[row["Crop"]]
        else:
            prediction = overall_median

        predictions.append(prediction)

    return np.asarray(predictions)


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH, low_memory=False)

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN], errors="coerce"
    )

    df = df.dropna(subset=["Year", TARGET_COLUMN, "Crop"]).copy()
    df = df[df[TARGET_COLUMN] > 0].copy()

    train = df[df["Year"] <= TRAIN_END_YEAR].copy()

    test = df[
        df["Year"].between(TEST_START_YEAR, TEST_END_YEAR)
    ].copy()

    if train.empty or test.empty:
        raise ValueError("Training or test split is empty.")

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]

    X_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMN]

    print("\n=== DATA SPLIT ===")
    print(f"Training rows: {len(train):,}")
    print(
        f"Training years: {int(train['Year'].min())}-"
        f"{int(train['Year'].max())}"
    )
    print(f"Test rows: {len(test):,}")
    print(f"Test years: {TEST_START_YEAR}-{TEST_END_YEAR}")

    # Load the existing baseline artifact without modifying it.
    print("\nLoading existing Random Forest artifact...")
    existing_rf = joblib.load(MODEL_ARTIFACT_PATH)

    models = {
        "Existing Random Forest": existing_rf,

        "Extra Trees baseline (300 trees)": build_pipeline(
            ExtraTreesRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),

        "Tuned Extra Trees": build_pipeline(
            ExtraTreesRegressor(
                n_estimators=100,
                min_samples_leaf=2,
                max_features=1.0,
                max_depth=None,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),

        "Tuned LightGBM": build_pipeline(
            LGBMRegressor(
                objective="regression",
                n_estimators=600,
                learning_rate=0.03,
                num_leaves=63,
                min_child_samples=10,
                reg_lambda=0.0,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=-1,
            )
        ),
    }

    results = []
    crop_results = []
    prediction_data = test[
        ["Year", "State Name", "Dist Name", "Crop", TARGET_COLUMN]
    ].copy()

    for name, model in models.items():
        print(f"\nTraining/evaluating: {name}")

        if name != "Existing Random Forest":
            model.fit(X_train, y_train)

        predictions = np.asarray(model.predict(X_test)).reshape(-1)
        metrics = calculate_metrics(y_test.to_numpy(), predictions)

        results.append({
            "Model": name,
            **metrics,
        })

        prediction_data[name] = predictions

        print(f"MAE:        {metrics['MAE']:.4f}")
        print(f"RMSE:       {metrics['RMSE']:.4f}")
        print(f"R2:         {metrics['R2']:.4f}")
        print(f"Mean error: {metrics['Mean_Error']:.4f}")

        for crop in sorted(test["Crop"].unique()):
            mask = test["Crop"].eq(crop).to_numpy()

            crop_metrics = calculate_metrics(
                y_test.to_numpy()[mask],
                predictions[mask],
            )

            crop_results.append({
                "Model": name,
                "Crop": crop,
                "Rows": int(mask.sum()),
                **crop_metrics,
            })

    # Baseline that predicts the training median for each crop/state.
    median_predictions = crop_state_median_predictions(train, test)
    median_name = "Crop-State Median"

    median_metrics = calculate_metrics(
        y_test.to_numpy(), median_predictions
    )

    results.append({
        "Model": median_name,
        **median_metrics,
    })

    prediction_data[median_name] = median_predictions

    for crop in sorted(test["Crop"].unique()):
        mask = test["Crop"].eq(crop).to_numpy()

        crop_metrics = calculate_metrics(
            y_test.to_numpy()[mask],
            median_predictions[mask],
        )

        crop_results.append({
            "Model": median_name,
            "Crop": crop,
            "Rows": int(mask.sum()),
            **crop_metrics,
        })

    results_df = pd.DataFrame(results).sort_values("MAE")
    crop_df = pd.DataFrame(crop_results)

    output_dir = PROJECT_ROOT / "reports" / "model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "test_model_comparison.csv"
    crop_path = output_dir / "test_crop_comparison.csv"
    predictions_path = output_dir / "test_model_predictions.csv"

    results_df.to_csv(results_path, index=False)
    crop_df.to_csv(crop_path, index=False)
    prediction_data.to_csv(predictions_path, index=False)

    print("\n" + "=" * 65)
    print("FINAL TEST COMPARISON (2015-2017)")
    print("=" * 65)
    print(results_df.round(4).to_string(index=False))

    print("\n=== TEST MAE BY CROP ===")
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
    print(predictions_path)

    print(
        "\nThe existing model artifact was not overwritten. "
        "The test set was used only for evaluation."
    )


if __name__ == "__main__":
    main()
