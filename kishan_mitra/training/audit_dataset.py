
import sys
from pathlib import Path

# Support direct execution from C:\sih
PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

import pandas as pd

from kishan_mitra.config import DATASET_PATH, TARGET_COLUMN


def audit_dataset():
    """Inspect dataset structure and potential modeling risks."""
    df = pd.read_csv(DATASET_PATH, low_memory=False)

    print("\n=== DATA STRUCTURE ===")
    print(f"Rows: {len(df)}")
    print(f"Years: {df['Year'].min()} to {df['Year'].max()}")
    print(f"Unique states: {df['State Name'].nunique()}")
    print(f"Unique districts: {df['Dist Name'].nunique()}")
    print(f"Unique crops: {df['Crop'].nunique()}")

    print("\n=== CROP DISTRIBUTION ===")
    print(df["Crop"].value_counts().to_string())

    print("\n=== ROWS PER YEAR ===")
    print(df.groupby("Year").size().describe().to_string())

    print("\n=== TARGET SUMMARY BY CROP ===")
    print(
        df.groupby("Crop")[TARGET_COLUMN]
        .agg(["count", "min", "median", "mean", "max"])
        .round(2)
        .to_string()
    )

    print("\n=== POSSIBLE KEY DUPLICATES ===")
    key_columns = ["Dist Code", "Year", "Crop"]
    duplicate_mask = df.duplicated(subset=key_columns, keep=False)

    print(
        "Rows sharing a district-year-crop key:",
        int(duplicate_mask.sum()),
    )

    if duplicate_mask.any():
        print(
            df.loc[duplicate_mask, key_columns]
            .sort_values(key_columns)
            .head(20)
            .to_string(index=False)
        )

    print("\n=== NUMERIC FEATURE CORRELATIONS ===")
    numeric = df.select_dtypes(include="number")
    correlations = (
        numeric.corr()[TARGET_COLUMN]
        .drop(TARGET_COLUMN)
        .dropna()
        .sort_values(key=lambda values: values.abs(), ascending=False)
    )
    print(correlations.head(15).round(4).to_string())

    print("\n=== NUTRIENT TOTAL FORMULA CHECK ===")
    formula_checks = [
        ("Total_N_kg", "N_req_kg_per_ha"),
        ("Total_P_kg", "P_req_kg_per_ha"),
        ("Total_K_kg", "K_req_kg_per_ha"),
    ]

    for total_col, rate_col in formula_checks:
        expected = df[rate_col] * df["Area_ha"]
        actual = df[total_col]

        difference = (actual - expected).abs()
        tolerance = 1e-5 * expected.abs().clip(lower=1)

        match_rate = (difference <= tolerance).mean()

        print(
            f"{total_col} vs {rate_col} * Area_ha: "
            f"{match_rate:.2%} rows match"
        )

    print("\n=== STATE CODE CONSISTENCY ===")
    state_mapping_counts = df.groupby("State Code")["State Name"].nunique()
    inconsistent_codes = state_mapping_counts[state_mapping_counts > 1]

    print(
        "State codes associated with multiple names:",
        len(inconsistent_codes),
    )

    if not inconsistent_codes.empty:
        print(inconsistent_codes.to_string())

    print("\n=== AUDIT COMPLETE ===")


if __name__ == "__main__":
    audit_dataset()
