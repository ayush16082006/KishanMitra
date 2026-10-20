
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from kishan_mitra.config import DATASET_PATH, TARGET_COLUMN


def main():
    df = pd.read_csv(DATASET_PATH, low_memory=False)

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN], errors="coerce"
    )

    df = df.dropna(subset=["Year", TARGET_COLUMN, "Crop"])

    print("\n=== YIELD DISTRIBUTION BY CROP ===")

    crop_stats = df.groupby("Crop")[TARGET_COLUMN].agg(
        ["count", "min", "median", "mean", "max", "std"]
    )

    print(crop_stats.round(2).to_string())

    print("\n=== YIELD PERCENTILES BY CROP ===")

    percentiles = (
        df.groupby("Crop")[TARGET_COLUMN]
        .quantile([0.50, 0.75, 0.90, 0.95, 0.99, 0.999])
        .unstack()
    )

    percentiles.columns = [
        "P50", "P75", "P90", "P95", "P99", "P99.9"
    ]

    print(percentiles.round(2).to_string())

    print("\n=== YEARLY YIELD SUMMARY ===")

    yearly = df.groupby("Year")[TARGET_COLUMN].agg(
        ["count", "median", "mean", "max"]
    )

    print(yearly.round(2).to_string())

    print("\n=== TOP 20 HIGHEST YIELD RECORDS ===")

    columns = [
        "Year",
        "State Name",
        "Dist Name",
        "Crop",
        "Area_ha",
        TARGET_COLUMN,
    ]

    print(
        df.nlargest(20, TARGET_COLUMN)[columns]
        .to_string(index=False)
    )

    print("\n=== ANALYSIS COMPLETE ===")


if __name__ == "__main__":
    main()
