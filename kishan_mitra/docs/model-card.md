# Kishan Mitra — Model Card

## 1. Model Overview

| Field | Description |
|---|---|
| Project | Kishan Mitra |
| Task | Crop-yield regression |
| Target | `Yield_kg_per_ha` |
| Output unit | Kilograms per hectare (kg/ha) |
| Production model | Random Forest regression pipeline |
| Candidate model | Extra Trees regression pipeline |
| Framework | scikit-learn |
| Serialization | Joblib |
| Application | Streamlit |
| Status | Random Forest deployed; Extra Trees experimental |

Kishan Mitra estimates crop yield from selected agricultural, location, and environmental features. The model is intended to support preliminary agricultural planning and is not designed to guarantee actual harvest outcomes.

## 2. Intended Use

### Appropriate uses

- Exploring approximate yield estimates for supported crops and locations.
- Demonstrating an end-to-end machine-learning application.
- Supporting preliminary agricultural planning alongside other information.
- Comparing model performance across candidate algorithms.
- Studying the effect of data quality, feature design, and model selection on regression performance.

### Out-of-scope uses

The model should not be used as the sole basis for:

- Guaranteed yield, income, or profit forecasts.
- Agricultural loans, insurance decisions, or other high-impact eligibility decisions.
- Prescribing exact fertilizer or pesticide quantities.
- Making major farming investments without independent validation.
- Claiming a guaranteed yield increase from AI-generated advice.

## 3. Dataset

**Dataset file:** `Datasets/Custom_Crops_yield_Historical_Dataset.csv`

The recorded dataset validation report contained **50,765 rows and 20 columns**.

The available columns include:

- Crop and administrative location identifiers.
- State, district, and year information.
- Yield and cultivated-area information.
- Nitrogen, phosphorus, and potassium requirement and total-quantity fields.
- Temperature, humidity, soil pH, rainfall, wind speed, and solar radiation.

The target is `Yield_kg_per_ha`.

The complete dataset schema should not be interpreted as the model's input schema. The prediction pipeline uses a selected feature set, which must be verified against its metadata and training configuration.

### Data provenance

The dataset filename and schema are documented in the repository. Its original publisher, collection methodology, licensing, completeness, and representativeness should be verified and documented from the original source before the project is used for consequential decisions.

### Data quality considerations

- Historical observations may contain missing values, inconsistent reporting, or extreme yields.
- Crop and location coverage may be uneven.
- The dataset may not represent recent climate patterns, farming practices, or local conditions.
- The meaning and reliability of individual environmental fields should be confirmed against the source documentation.
- Dataset row counts can change if the source file is updated.

## 4. Features and Preprocessing

The application collects crop, location, soil, and environmental information. The model's exact feature columns and transformations are defined by the saved pipeline and its associated configuration.

The application and training pipeline must agree on:

- Feature names and ordering.
- Categorical-value handling.
- Missing-value imputation.
- Numeric input types.
- Scaling or encoding transformations, where applicable.
- Handling of previously unseen categories.

Preprocessing should be fitted on training data and serialized with the estimator where appropriate. Inference must use the fitted transformations rather than refitting them on user inputs.

Fields present in the dataset are not automatically valid model features. Any feature unavailable or unreliable at the time of prediction must be evaluated carefully for inference-time suitability and potential leakage.

## 5. Evaluation Methodology

The project evaluates regression models using:

- **MAE — Mean Absolute Error:** average absolute prediction error, expressed in kg/ha.
- **RMSE — Root Mean Squared Error:** penalizes larger errors more strongly, expressed in kg/ha.
- **R² — Coefficient of Determination:** measures performance relative to a mean-based reference under the evaluation sample; it is not a percentage accuracy score.

The recorded model-comparison experiment used a held-out test period of **2015–2017**, containing **2,918 rows**. These years were not used for fitting or hyperparameter selection in that comparison.

Hyperparameter tuning used chronological cross-validation confined to the training period. The documented tuning folds progressed through historical periods, with later years reserved for validation and final testing.

Evaluation should be reproduced using the documented scripts and the corresponding dataset version before the results are treated as current.

## 6. Recorded Held-Out Test Results

The following results are from the repository's recorded model-comparison experiment.

| Model | MAE (kg/ha) | RMSE (kg/ha) | R² |
|---|---:|---:|---:|
| Existing Random Forest | 454.09 | 821.56 | 0.6737 |
| Extra Trees, 300 trees | 441.89 | 810.44 | 0.6825 |
| Tuned Extra Trees | 442.03 | 811.45 | 0.6817 |
| Tuned LightGBM | 457.22 | 820.90 | 0.6742 |
| Crop-state median baseline | 868.36 | 1,362.17 | 0.1031 |

### Interpretation

- Extra Trees with 300 trees achieved the lowest overall test MAE among the listed models.
- Its MAE was approximately 2.69% lower than the existing Random Forest's MAE in this experiment.
- The Extra Trees model also recorded lower RMSE and higher R² than the existing Random Forest.
- Hyperparameter tuning did not improve the reported test metrics over the original Extra Trees configuration.
- Tuned LightGBM had a slightly higher R² than the existing Random Forest, but a higher MAE.

These results indicate that Extra Trees was the leading overall candidate in this particular comparison. They do not establish that it is best for every crop or every future prediction.

The reported MAE values describe average absolute error, not the maximum possible error. Individual predictions can be substantially less accurate.

## 7. Crop-Level Performance

The recorded test comparison showed variation by crop.

| Crop | Random Forest MAE (kg/ha) | Extra Trees MAE (kg/ha) |
|---|---:|---:|
| Chickpea | 255.15 | 260.31 |
| Cotton | 222.67 | 219.93 |
| Maize | 803.67 | 777.92 |
| Rice | 411.91 | 391.79 |

Extra Trees performed better on cotton, maize, and rice in this comparison, but Random Forest performed better on chickpea.

This variation is important. Aggregate metrics can conceal weaker performance for individual crops, and crop-level sample sizes and error distributions should be examined before making crop-specific claims.

## 8. Production Model and Candidate Policy

### Production: Random Forest

The deployed application explicitly selects `random_forest`, which loads the existing pipeline from:

`models/crop_yield_pipeline.joblib`

The production artifact and its metadata should remain consistent with the deployed dependency versions and input schema.

### Candidate: Extra Trees

The experimental artifact is:

`models/crop_yield_extratrees_candidate.joblib`

Its recorded test MAE is lower than the existing Random Forest's MAE. However, a candidate's evaluation result is only one part of the promotion decision.

Before promotion, the candidate must pass:

1. Reproducible evaluation.
2. Crop-level and error-distribution review.
3. Input and preprocessing compatibility checks.
4. Prediction unit and output validation.
5. Automated and end-to-end tests.
6. Deployment artifact and memory checks.
7. Explicit documentation of the final model-selection decision.

Until these checks are complete, the Extra Trees candidate must not be described as the model powering the deployed application.

## 9. Limitations and Risks

### Historical-data limitations

Historical relationships between weather, location, crop, and yield may not hold under new climate conditions, new farming practices, or unusual seasons.

### Geographic and crop variation

Performance can differ by crop, state, district, or data density. Aggregate test results do not guarantee equivalent accuracy for an individual location.

### Feature availability

A prediction may rely on environmental values that are not representative of the entire growing season. Current weather alone cannot capture future rainfall, pests, irrigation, soil fertility, or other important determinants of yield.

### Error magnitude

The recorded test MAE for the existing Random Forest was 454.09 kg/ha, with RMSE of 821.56 kg/ha. The larger RMSE indicates that some larger errors affect overall performance. These metrics should be communicated when interpreting predictions.

### AI Advisor limitations

The AI Advisor is a separate LLM-powered component. Its recommendations may be incorrect, incomplete, or insufficiently specific to local conditions. The regression prediction does not validate the agronomic correctness of generated advice.

### Deployment limitations

Saved models require compatible Python and machine-learning library versions. Large artifacts also impose storage and memory requirements. A model that works locally must still pass deployment-specific tests.

## 10. Fairness, Safety, and Responsible Use

- Display predictions as estimates, not facts or guarantees.
- Communicate units clearly.
- Avoid claiming guaranteed percentage improvements in yield or profit.
- Avoid inventing fertilizer doses, pesticide instructions, or local agricultural facts.
- Encourage verification through local agricultural extension services or qualified agronomists.
- Record failures and investigate unusually large prediction errors.
- Re-evaluate performance when the dataset, model, feature schema, or target population changes.

## 11. Monitoring and Future Evaluation

Recommended future work includes:

- Re-evaluating the model on newer, independent data when available.
- Reporting error distributions and residual plots.
- Tracking performance by crop and geographic region where sample sizes permit.
- Investigating underprediction and overprediction patterns.
- Verifying data provenance, licensing, and feature semantics.
- Testing the complete deployed workflow after model or dependency changes.
- Maintaining a reproducible record of dataset versions, feature schema, training parameters, and metrics.

## 12. References Within the Repository

- Dataset: `Datasets/Custom_Crops_yield_Historical_Dataset.csv`
- Model artifacts: `models/`
- Training and evaluation scripts: `training/`
- Model comparison evidence: `reports/model_comparison/`
- Architecture: `docs/architecture.md`
- Prediction tests: `tests/test_prediction.py`

**Model-card status:** This document records the current model setup and available experimental results. Dataset provenance, detailed feature metadata, and future evaluation results should be updated when verified evidence becomes available.
