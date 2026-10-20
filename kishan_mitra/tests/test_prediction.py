import unittest

import numpy as np
import pandas as pd

from kishan_mitra.config import DATASET_PATH
from kishan_mitra.prediction import (
    FEATURE_COLUMNS,
    predict_one,
    predict_yield,
)


class TestPredictionService(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Use a real dataset row as a realistic test input.
        df = pd.read_csv(DATASET_PATH, low_memory=False)
        cls.sample = df[FEATURE_COLUMNS].iloc[0].to_dict()

    def test_random_forest_prediction(self):
        prediction = predict_one(
            self.sample,
            model_name="random_forest",
        )

        self.assertTrue(np.isfinite(prediction))
        self.assertGreater(prediction, 0)

    def test_extra_trees_candidate_prediction(self):
        prediction = predict_one(
            self.sample,
            model_name="extra_trees_candidate",
        )

        self.assertTrue(np.isfinite(prediction))
        self.assertGreater(prediction, 0)

    def test_unseen_categories(self):
        sample = self.sample.copy()
        sample["State Name"] = "Unseen_Test_State"
        sample["Dist Name"] = "Unseen_Test_District"
        sample["Crop"] = "Unseen_Test_Crop"

        prediction = predict_one(
            sample,
            model_name="extra_trees_candidate",
        )

        self.assertTrue(np.isfinite(prediction))

    def test_missing_values(self):
        sample = self.sample.copy()
        sample["State Name"] = None
        sample["Temperature_C"] = np.nan
        sample["Rainfall_mm"] = np.nan

        prediction = predict_one(
            sample,
            model_name="extra_trees_candidate",
        )

        self.assertTrue(np.isfinite(prediction))

    def test_batch_predictions(self):
        batch = pd.DataFrame([self.sample, self.sample])

        predictions = predict_yield(
            batch,
            model_name="extra_trees_candidate",
        )

        self.assertEqual(len(predictions), 2)
        self.assertTrue(np.isfinite(predictions).all())

    def test_missing_required_column_is_rejected(self):
        sample = self.sample.copy()
        del sample["Crop"]

        with self.assertRaisesRegex(
            ValueError,
            "Missing required feature columns",
        ):
            predict_one(
                sample,
                model_name="extra_trees_candidate",
            )

    def test_unknown_model_name_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            predict_one(
                self.sample,
                model_name="unknown_model",
            )

    def test_empty_batch_is_rejected(self):
        empty_batch = pd.DataFrame(columns=FEATURE_COLUMNS)

        with self.assertRaisesRegex(
            ValueError,
            "at least one row",
        ):
            predict_yield(
                empty_batch,
                model_name="extra_trees_candidate",
            )

    def test_infinite_numeric_value_is_rejected(self):
        sample = self.sample.copy()
        sample["Temperature_C"] = np.inf

        with self.assertRaisesRegex(
            ValueError,
            "infinity",
        ):
            predict_one(
                sample,
                model_name="extra_trees_candidate",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)