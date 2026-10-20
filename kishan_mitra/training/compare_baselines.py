
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from kishan_mitra.config import DATASET_PATH, TARGET_COLUMN
from training.model_config import (
    TRAIN_END_YEAR,
    TEST_START_YEAR,
    TEST_END_YEAR,
)


def metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
    }


def main():
    df = pd.read_csv(DATASET_PATH, low_memory=False)

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN], errors="coerce"
    )

    df = df.dropna(subset=["Year", TARGET_COLUMN, "Crop"])
    df = df[df[TARGET_COLUMN] > 0]

    train = df[df["Year"] <= TRAIN_END_YEAR].copy()
    test = df[
        df["Year"].between(TEST_START_YEAR, TEST_END_YEAR)
    ].copy()

    if train.empty or test.empty:
        raise ValueError("Training or test split is empty.")

    # Baseline 1: overall historical median.
    overall_median = train[TARGET_COLUMN].median()
    test["Overall_Median"] = overall_median

    # Baseline 2: historical median for each crop.
    crop_medians = train.groupby("Crop")[TARGET_COLUMN].median()
    test["Crop_Median"] = (
        test["Crop"].map(crop_medians).fillna(overall_median)
    )

    # Baseline 3: historical median for each crop and state.
    crop_state_medians = train.groupby(
        ["Crop", "State Name"]
    )[TARGET_COLUMN].median()

    state_keys = pd.MultiIndex.from_frame(
        test[["Crop", "State Name"]]
    )

    test["Crop_State_Median"] = (
        crop_state_medians.reindex(state_keys).to_numpy()
    )

    test["Crop_State_Median"] = test["Crop_State_Median"].fillna(
        test["Crop_Median"]
    )

    actual = test[TARGET_COLUMN].to_numpy()

    print("\n=== TEST BASELINE COMPARISON ===")
    print(f"Training years: through {TRAIN_END_YEAR}")
    print(f"Test years: {TEST_START_YEAR}-{TEST_END_YEAR}")
    print(f"Test rows: {len(test):,}")

    for name in [
        "Overall_Median",
        "Crop_Median",
        "Crop_State_Median",
    ]:
        result = metrics(actual, test[name].to_numpy())
        print(f"\n{name}")
        for metric_name, value in result.items():
            print(f"{metric_name}: {value:,.4f}")

    print("\n=== BASELINE MAE BY CROP ===")
    rows = []

    for crop, group in test.groupby("Crop"):
        actual_crop = group[TARGET_COLUMN].to_numpy()

        row = {"Crop": crop}
        for name in [
            "Overall_Median",
            "Crop_Median",
            "Crop_State_Median",
        ]:
            row[name] = mean_absolute_error(
                actual_crop,
                group[name].to_numpy(),
            )

        rows.append(row)

    print(
        pd.DataFrame(rows)
        .set_index("Crop")
        .round(2)
        .to_string()
    )

    print("\n=== COMPARISON COMPLETE ===")


if __name__ == "__main__":
    main()
