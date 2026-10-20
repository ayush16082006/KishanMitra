import sys
from pathlib import Path

# Add C:\sih to Python's import path for direct script execution.
PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

import pandas as pd

from kishan_mitra.config import DATASET_PATH, TARGET_COLUMN



def validate_dataset(dataset_path: Path = None) -> dict:
    """Inspect the crop-yield dataset without modifying the source file."""

    path = Path(dataset_path or DATASET_PATH)

    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, low_memory=False)

    if df.empty:
        raise ValueError("The dataset contains no rows.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing. "
            f"Available columns: {df.columns.tolist()}"
        )

    target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")

    report = {
        "file": str(path.resolve()),
        "rows": int(len(df)),
        "columns_count": int(len(df.columns)),
        "columns": df.columns.tolist(),
        "dtypes": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },
        "missing_values": {
            column: int(count)
            for column, count in df.isna().sum().items()
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "target": {
            "column": TARGET_COLUMN,
            "non_numeric_or_missing": int(target.isna().sum()),
            "zero_or_negative": int((target <= 0).sum()),
            "minimum": (
                float(target.min()) if target.notna().any() else None
            ),
            "median": (
                float(target.median()) if target.notna().any() else None
            ),
            "maximum": (
                float(target.max()) if target.notna().any() else None
            ),
        },
    }

    return report


def print_report(report: dict) -> None:
    """Print the validation report in a readable format."""

    print("\n=== DATASET VALIDATION REPORT ===")
    print("File:", report["file"])
    print("Rows:", report["rows"])
    print("Columns:", report["columns_count"])
    print("Column names:", report["columns"])
    print("\nData types:")
    for column, dtype in report["dtypes"].items():
        print(f"  {column}: {dtype}")

    print("\nMissing values:")
    for column, count in report["missing_values"].items():
        if count > 0:
            print(f"  {column}: {count}")

    print("Duplicate rows:", report["duplicate_rows"])
    print("\nTarget summary:")
    for key, value in report["target"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    print_report(validate_dataset())
