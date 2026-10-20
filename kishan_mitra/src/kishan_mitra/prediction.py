
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from kishan_mitra.config import (
    MODELS_DIR,
    MODEL_ARTIFACT_PATH,
    MODEL_METADATA_PATH,
    TARGET_COLUMN,
)
from training.model_config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
DEFAULT_MODEL_NAME = "random_forest"

CANDIDATE_MODEL_PATH = (
    MODELS_DIR / "crop_yield_extratrees_candidate.joblib"
)
CANDIDATE_METADATA_PATH = (
    MODELS_DIR / "crop_yield_extratrees_candidate_metadata.json"
)

MODEL_PATHS = {
    "random_forest": MODEL_ARTIFACT_PATH,
    "extra_trees_candidate": CANDIDATE_MODEL_PATH,
}

METADATA_PATHS = {
    "random_forest": MODEL_METADATA_PATH,
    "extra_trees_candidate": CANDIDATE_METADATA_PATH,
}


def load_model(model_name=DEFAULT_MODEL_NAME):
    """Load a saved model pipeline without changing any artifacts."""
    if model_name not in MODEL_PATHS:
        allowed = ", ".join(MODEL_PATHS)
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {allowed}"
        )

    model_path = MODEL_PATHS[model_name]

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}"
        )

    model = joblib.load(model_path)

    # Check feature metadata when a metadata file is available.
    metadata_path = METADATA_PATHS[model_name]
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        recorded_features = metadata.get("feature_columns")
        if (
            recorded_features is not None
            and recorded_features != FEATURE_COLUMNS
        ):
            raise ValueError(
                f"Feature configuration mismatch in {metadata_path}"
            )

    return model


def predict_yield(input_data, model_name=DEFAULT_MODEL_NAME):
    """
    Predict crop yield in kg/ha.

    input_data may be a dictionary or a pandas DataFrame.
    The input must contain all configured features. Extra fields are ignored.
    Missing feature values are passed to the saved pipeline's imputers.
    """
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        input_df = input_data.copy()
    else:
        raise TypeError(
            "input_data must be a dictionary or pandas DataFrame."
        )

    if input_df.empty:
        raise ValueError("Input data must contain at least one row.")

    missing_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in input_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required feature columns: "
            + ", ".join(missing_columns)
        )

    # Select and order features exactly as they were defined during training.
    X = input_df.loc[:, FEATURE_COLUMNS].copy()

    # Convert numeric features to numbers; invalid values become missing.
    for column in NUMERIC_FEATURES:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    # Reject infinities and invalid numeric values that would otherwise
    # reach preprocessing. NaN is allowed because the pipeline imputes it.
    numeric_values = X[NUMERIC_FEATURES].to_numpy(dtype=float)
    if np.isinf(numeric_values).any():
        raise ValueError(
            "Numeric features cannot contain positive or negative infinity."
        )

    model = load_model(model_name)
    predictions = np.asarray(model.predict(X), dtype=float)

    if predictions.ndim != 1 or len(predictions) != len(X):
        raise RuntimeError("The model returned an unexpected output shape.")

    if not np.isfinite(predictions).all():
        raise RuntimeError("The model returned a non-finite prediction.")

    return predictions


def predict_one(input_data, model_name=DEFAULT_MODEL_NAME):
    """Return one yield prediction as a Python float."""
    predictions = predict_yield(input_data, model_name=model_name)

    if len(predictions) != 1:
        raise ValueError(
            "predict_one expects exactly one input row."
        )

    return float(predictions[0])
