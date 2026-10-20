
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from kishan_mitra.config import (
    DATASET_PATH,
    MODEL_ARTIFACT_PATH,
    TARGET_COLUMN,
)
from training.model_config import (
    FEATURE_COLUMNS,
    TEST_START_YEAR,
    TEST_END_YEAR,
)


def calculate_metrics(actual, predicted):
    return {
        "rows": len(actual),
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": (
            r2_score(actual, predicted)
            if len(actual) >= 2
            else float("nan")
        ),
        "Mean_Error": np.mean(predicted - actual),
    }


def main():
    if not MODEL_ARTIFACT_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_ARTIFACT_PATH}. "
            "Run training/train_model.py first."
        )

    df = pd.read_csv(DATASET_PATH, low_memory=False)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN], errors="coerce"
    )

    test = df[
        df["Year"].between(TEST_START_YEAR, TEST_END_YEAR)
        & df[TARGET_COLUMN].notna()
        & (df[TARGET_COLUMN] > 0)
    ].copy()

    if test.empty:
        raise ValueError("No valid rows found for the test period.")

    model = joblib.load(MODEL_ARTIFACT_PATH)

    X_test = test[FEATURE_COLUMNS]
    actual = test[TARGET_COLUMN].to_numpy()
    predicted = model.predict(X_test)

    test["Actual_Yield"] = actual
    test["Predicted_Yield"] = predicted
    test["Absolute_Error"] = np.abs(actual - predicted)
    test["Prediction_Error"] = predicted - actual

    print("\n=== OVERALL TEST METRICS ===")
    for name, value in calculate_metrics(actual, predicted).items():
        if name == "rows":
            print(f"{name}: {value:,}")
        else:
            print(f"{name}: {value:,.4f}")

    print("\n=== METRICS BY CROP ===")
    rows = []

    for crop, group in test.groupby("Crop"):
        metrics = calculate_metrics(
            group["Actual_Yield"].to_numpy(),
            group["Predicted_Yield"].to_numpy(),
        )
        metrics["Crop"] = crop
        rows.append(metrics)

    crop_metrics = pd.DataFrame(rows).set_index("Crop")
    print(crop_metrics.round(3).to_string())

    print("\n=== WORST 10 PREDICTIONS BY ABSOLUTE ERROR ===")
    columns = [
        "Year",
        "State Name",
        "Dist Name",
        "Crop",
        "Actual_Yield",
        "Predicted_Yield",
        "Absolute_Error",
    ]

    print(
        test.nlargest(10, "Absolute_Error")[columns]
        .round(2)
        .to_string(index=False)
    )

    report_dir = PROJECT_ROOT / "reports" / "model_comparison"
    report_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = report_dir / "test_predictions.csv"
    metrics_path = report_dir / "crop_metrics.csv"

    test.to_csv(predictions_path, index=False)
    crop_metrics.to_csv(metrics_path)

    print("\n=== EVALUATION FILES SAVED ===")
    print(f"Predictions: {predictions_path}")
    print(f"Crop metrics: {metrics_path}")


if __name__ == "__main__":
    main()
