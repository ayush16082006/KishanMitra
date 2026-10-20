
from kishan_mitra.config import TARGET_COLUMN, RANDOM_STATE, CV_FOLDS

# Features provisionally available before harvest.
# Nutrient requirements and their derived totals are excluded.
CATEGORICAL_FEATURES = [
    "State Name",
    "Dist Name",
    "Crop",
]

NUMERIC_FEATURES = [
    "Year",
    "Temperature_C",
    "Humidity_%",
    "pH",
    "Rainfall_mm",
    "Wind_Speed_m_s",
    "Solar_Radiation_MJ_m2_day",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Chronological split: never randomly mix future records into training.
TRAIN_END_YEAR = 2012

VALIDATION_START_YEAR = 2013
VALIDATION_END_YEAR = 2014

TEST_START_YEAR = 2015
TEST_END_YEAR = 2017

# Model settings
RANDOM_STATE = RANDOM_STATE
CV_FOLDS = CV_FOLDS
