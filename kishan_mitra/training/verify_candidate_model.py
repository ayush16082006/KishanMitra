
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from kishan_mitra.config import (
    DATASET_PATH,
    MODEL_ARTIFACT_PATH,
)
from model_config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_MODEL_PATH = (
    PROJECT_ROOT / "models" / "crop_yield_extratrees_candidate.joblib"
)
CANDIDATE_METADATA_PATH = (
    PROJECT_ROOT / "models" / "crop_yield_extratrees_candidate_metadata.json"
)

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def main():
    print("=== EXTRA TREES CANDIDATE VERIFICATION ===")

    # 1. Check that the separate artifacts exist.
    if not CANDIDATE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Candidate model not found: {CANDIDATE_MODEL_PATH}"
        )

    if not CANDIDATE_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Candidate metadata not found: {CANDIDATE_METADATA_PATH}"
        )

    print("[PASS] Candidate artifact and metadata exist.")

    # 2. Load the saved pipeline and metadata.
    model = joblib.load(CANDIDATE_MODEL_PATH)

    with open(CANDIDATE_METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    if metadata["model_name"] != "Extra Trees Regressor":
        raise AssertionError("Unexpected model name in metadata.")

    if metadata["feature_columns"] != FEATURE_COLUMNS:
        raise AssertionError(
            "Metadata feature columns do not match model configuration."
        )

    print("[PASS] Saved model and metadata loaded.")
    print("[PASS] Metadata feature configuration matches.")

    # 3. Build representative input from a real dataset row.
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    sample = df[FEATURE_COLUMNS].iloc[[0]].copy()

    # Case A: ordinary input.
    normal_input = sample.copy()

    # Case B: categories not encountered during training.
    unseen_input = sample.copy()
    unseen_input.loc[:, "State Name"] = "Unseen_Test_State"
    unseen_input.loc[:, "Dist Name"] = "Unseen_Test_District"
    unseen_input.loc[:, "Crop"] = "Unseen_Test_Crop"

    # Case C: missing values in categorical and numeric features.
    missing_input = sample.copy()
    missing_input.loc[:, "State Name"] = np.nan
    missing_input.loc[:, "Temperature_C"] = np.nan
    missing_input.loc[:, "Rainfall_mm"] = np.nan

    test_inputs = pd.concat(
        [normal_input, unseen_input, missing_input],
        ignore_index=True,
    )

    # 4. Predict with the loaded artifact.
    predictions = model.predict(test_inputs)

    if len(predictions) != 3:
        raise AssertionError("Expected three predictions.")

    if not np.isfinite(predictions).all():
        raise AssertionError("Model returned non-finite predictions.")

    print("[PASS] Normal input produces a finite prediction.")
    print("[PASS] Unseen categories are handled.")
    print("[PASS] Missing input values are handled.")

    print("\n=== SAMPLE PREDICTIONS ===")
    for label, prediction in zip(
        ["Normal input", "Unseen categories", "Missing values"],
        predictions,
    ):
        print(f"{label}: {prediction:.2f} kg/ha")

    # 5. Confirm that the existing Random Forest artifact still exists.
    if MODEL_ARTIFACT_PATH.exists():
        print("\n[PASS] Existing Random Forest artifact still exists.")
        print(f"Random Forest: {MODEL_ARTIFACT_PATH}")
    else:
        print(
            "\n[WARNING] Configured Random Forest artifact was not found "
            f"at: {MODEL_ARTIFACT_PATH}"
        )

    print(f"\nCandidate: {CANDIDATE_MODEL_PATH}")
    print("\nVERIFICATION COMPLETED.")


if __name__ == "__main__":
    main()
