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

    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN], errors="coerce"
    )

    numeric_columns = [
        "Year",
        "Area_ha",
        "N_req_kg_per_ha",
        "P_req_kg_per_ha",
        "K_req_kg_per_ha",
        "Total_N_kg",
        "Total_P_kg",
        "Total_K_kg",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=[TARGET_COLUMN, "Crop"])

    # Calculate an extreme-yield threshold separately for each crop.
    thresholds = df.groupby("Crop")[TARGET_COLUMN].quantile(0.999)

    df["Crop_P99_9"] = df["Crop"].map(thresholds)

    extreme = df[df[TARGET_COLUMN] > df["Crop_P99_9"]].copy()

    # Ratios are diagnostic only; they do not prove data leakage.
    extreme["Yield_to_N_Ratio"] = (
        extreme[TARGET_COLUMN] / extreme["N_req_kg_per_ha"]
    )
    extreme["Yield_to_P_Ratio"] = (
        extreme[TARGET_COLUMN] / extreme["P_req_kg_per_ha"]
    )
    extreme["Yield_to_K_Ratio"] = (
        extreme[TARGET_COLUMN] / extreme["K_req_kg_per_ha"]
    )

    columns = [
        "Year",
        "State Name",
        "Dist Name",
        "Crop",
        "Area_ha",
        TARGET_COLUMN,
        "Crop_P99_9",
        "N_req_kg_per_ha",
        "P_req_kg_per_ha",
        "K_req_kg_per_ha",
        "Total_N_kg",
        "Total_P_kg",
        "Total_K_kg",
        "Yield_to_N_Ratio",
        "Yield_to_P_Ratio",
        "Yield_to_K_Ratio",
    ]

    print("\n=== EXTREME YIELDS BY CROP ===")
    print(
        extreme.groupby("Crop")
        .size()
        .rename("Records above crop P99.9")
        .to_string()
    )

    print("\n=== EXTREME RECORD DETAILS ===")
    print(
        extreme.sort_values(TARGET_COLUMN, ascending=False)[columns]
        .head(30)
        .round(3)
        .to_string(index=False)
    )

    print("\n=== AUDIT COMPLETE ===")


if __name__ == "__main__":
    main()