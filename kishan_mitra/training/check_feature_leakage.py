
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from kishan_mitra.config import DATASET_PATH, TARGET_COLUMN


def main():
    df = pd.read_csv(DATASET_PATH, low_memory=False)

    nutrient_columns = [
        "N_req_kg_per_ha",
        "P_req_kg_per_ha",
        "K_req_kg_per_ha",
    ]

    print("\n=== NUTRIENT FEATURE INVESTIGATION ===")

    for column in nutrient_columns:
        correlation = df[column].corr(df[TARGET_COLUMN])

        # Check whether the yield-to-nutrient ratio is nearly constant.
        valid = df[column].notna() & df[column].ne(0)
        ratio = df.loc[valid, TARGET_COLUMN] / df.loc[valid, column]

        print(f"\nFeature: {column}")
        print(f"Correlation with yield: {correlation:.6f}")
        print(
            "Yield / nutrient ratio — "
            f"min: {ratio.min():.4f}, "
            f"median: {ratio.median():.4f}, "
            f"max: {ratio.max():.4f}"
        )

        print("\nCorrelation by crop:")
        print(
            df.groupby("Crop")
            .apply(
                lambda group: group[column].corr(group[TARGET_COLUMN]),
                include_groups=False,
            )
            .round(4)
            .to_string()
        )

    print("\n=== HIGHEST YIELD RECORDS ===")

    columns_to_show = [
        "Year",
        "State Name",
        "Dist Name",
        "Crop",
        "Area_ha",
        TARGET_COLUMN,
        *nutrient_columns,
    ]

    print(
        df.nlargest(10, TARGET_COLUMN)[columns_to_show]
        .to_string(index=False)
    )

    print("\n=== INVESTIGATION COMPLETE ===")


if __name__ == "__main__":
    main()
