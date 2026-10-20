**# Model Experiment Log — Kishan Mitra**



**\*\*Project:\*\*** Kishan Mitra — Crop Yield Prediction and AI Advisor  

**\*\*Document:\*\*** Model Experiment Log  

**\*\*Target variable:\*\*** \`Yield_kg_per_ha\`  

**\*\*Evaluation approach:\*\*** Chronological training, validation, and held-out testing  

**\*\*Random state:\*\*** 42  

**\*\*Status:\*\*** Extra Trees candidate verified; production promotion not yet authorized



**---**



**## 1. Purpose**



This document records the model-development process used to select a crop-yield regression model for Kishan Mitra.



The objective is not simply to maximize a metric on one dataset split. The objective is to select a model that generalizes to later years, avoids obvious target leakage, performs reasonably across supported crops, handles realistic input problems, and can be integrated into the application without damaging the existing working model.



The experiment process includes:



1\. Dataset validation and feature auditing.

2\. Selection of defensible model inputs.

3\. Baseline comparisons.

4\. Comparison of different regression algorithms.

5\. Hyperparameter tuning using chronological cross-validation.

6\. Evaluation on a separate, later-period test set.

7\. Saving and verifying a candidate model independently.

8\. Documenting limitations and conditions for production promotion.



The existing Random Forest pipeline remains preserved until the replacement candidate passes the required evaluation and end-to-end integration checks.



**## 2. Dataset Overview**



The dataset is located at:



\`Datasets/Custom_Crops_yield_Historical_Dataset.csv\`



\| Property | Value |

\|---|---:|

\| Total rows | 50,765 |

\| Total columns | 20 |

\| Historical years | 1966–2017 |

\| States | 20 |

\| Districts | 311 |

\| Crops | 4 |

\| Missing values in original audit | None |

\| Exact duplicate rows in original audit | None |

\| Missing target values | None |

\| Non-positive target values | None |



The supported crops are chickpea, cotton, maize, and rice.



**### 2.1 Target variable**



The prediction target is \`Yield_kg_per_ha\`, representing crop yield in kilograms per hectare.



The observed target distribution is highly variable. The overall target ranges from 0.28 to 58,363.64 kg/ha, and the median is approximately 922.08 kg/ha.



Extreme values require investigation rather than automatic deletion. Some observations may be valid, while others may reflect source-data or measurement issues.



**### 2.2 Crop-level target distribution**



\| Crop | Rows | Median yield | Mean yield | Maximum yield |

\|---|---:|---:|---:|---:|

\| Chickpea | 13,847 | 671.52 | 734.08 | 58,363.64 |

\| Cotton | 7,804 | 211.06 | 257.47 | 5,000.00 |

\| Maize | 14,136 | 1,300.24 | 1,606.04 | 21,428.57 |

\| Rice | 14,978 | 1,417.05 | 1,602.41 | 5,653.83 |



Yield distributions differ substantially between crops. This supports evaluating errors by crop rather than relying exclusively on a single overall score.



**### 2.3 Data provenance and limitations**



The dataset was reported as downloaded from Kaggle. However, the original listing or download source could not be independently verified during the project audit. Its provenance should therefore be recorded as unverified rather than represented as confirmed.



The dataset ends in 2017. It does not establish how well a model trained on this data will perform under current agricultural conditions, contemporary farming practices, or future weather patterns.



**## 3. Dataset Audit and Feature Selection**



**### 3.1 Data-quality checks**



The audit covered missing values, exact duplicates, target validity, duplicate district-year-crop keys, state-code consistency, nutrient-total calculations, yield distributions, and extreme observations.



The original audit found:



\- No missing values in the dataset.

\- No exact duplicate rows.

\- No missing, non-numeric, zero, or negative target values.

\- No duplicate \`Dist Code\`, \`Year\`, and \`Crop\` keys.

\- Consistent state codes with state names.

\- Nutrient totals consistent with their corresponding nutrient requirement multiplied by area in the audited rows.



These checks improve confidence in the dataset's internal consistency but do not prove that every observation is historically correct.



**### 3.2 Potential target leakage**



The nutrient requirement features showed extremely strong correlations with the target, including correlations of 1.0 for individual crops in the earlier analysis.



The following features were excluded from the model inputs:



\- \`N_req_kg_per_ha\`

\- \`P_req_kg_per_ha\`

\- \`K_req_kg_per_ha\`

\- \`Total_N_kg\`

\- \`Total_P_kg\`

\- \`Total_K_kg\`



These variables may be derived from the target or from a formula closely linked to it. The original data-generation process has not been confirmed, so this is a potential-leakage finding, not proof of leakage.



Excluding them is a conservative modeling decision intended to reduce the risk of obtaining deceptively strong evaluation metrics from information unavailable at prediction time.



**### 3.3 Other excluded variables**



\- \`Area_ha\`: historical district-level crop area may not represent an individual farmer's field.

\- \`Dist Code\` and \`State Code\`: redundant with the retained geographic names for this feature configuration.



**### 3.4 Final feature configuration**



**\*\*Categorical features\*\***



\- \`State Name\`

\- \`Dist Name\`

\- \`Crop\`



**\*\*Numeric features\*\***



\- \`Year\`

\- \`Temperature_C\`

\- \`Humidity\_%\`

\- \`pH\`

\- \`Rainfall_mm\`

\- \`Wind_Speed_m_s\`

\- \`Solar_Radiation_MJ_m2_day\`



**\*\*Target\*\***



\- \`Yield_kg_per_ha\`



This configuration aims to use geographic identity, crop type, historical year, and environmental or soil measurements while excluding the potentially target-derived nutrient fields.



**## 4. Experimental Design**



**### 4.1 Chronological data split**



The experiments use year-based splits rather than a random row split.



\| Dataset partition | Years | Rows |

\|---|---|---:|

\| Training | 1966–2012 | 45,925 |

\| Validation | 2013–2014 | 1,922 |

\| Held-out test | 2015–2017 | 2,918 |

\| Total | 1966–2017 | 50,765 |



The model is trained using earlier observations, evaluated during later validation years, and finally assessed on an even later test period.



This setup better reflects the challenge of predicting future agricultural outcomes from historical data. It also helps reduce the overly optimistic results that can occur when similar observations from multiple years are randomly distributed across training and testing.



**### 4.2 Evaluation metrics**



**\*\*Mean Absolute Error (MAE)\*\***



MAE is the average absolute difference between predicted and actual yield. It is measured in kg/ha, making it relatively easy to interpret.



**\*\*Root Mean Squared Error (RMSE)\*\***



RMSE penalizes large errors more strongly than MAE. It is useful when occasional very large prediction errors are important.



**\*\*R² score\*\***



R² measures performance relative to a mean-based reference. A higher score generally indicates better fit, but R² can be negative when a model performs worse than the reference predictor.



**\*\*Mean error\*\***



Mean error is the average signed difference between predictions and actual values. Under the convention used in the reports, a negative mean error indicates average underprediction.



No single metric completely describes model quality. Model selection considers MAE, RMSE, R², crop-level performance, and generalization to the held-out test period.



**## 5. Preprocessing Pipeline**



The tree-based regression pipelines use the following preprocessing:



**### 5.1 Categorical features**



\- Impute missing categorical values using the most frequent value.

\- Apply one-hot encoding.

\- Use \`handle_unknown="ignore"\` so unseen categories do not cause prediction failures.



**### 5.2 Numeric features**



\- Impute missing numeric values using the median.

\- Apply \`StandardScaler\`.



Scaling is generally unnecessary for tree-based models, but retaining a consistent preprocessing structure simplifies comparison with models that benefit from scaling. Any preprocessing statistics must be learned from the training data, not from validation or test data.



**### 5.3 Why use a pipeline?**



A saved pipeline keeps preprocessing and model prediction together. This reduces the chance that the application encodes categories, fills missing values, or orders features differently from the training process.



The candidate pipeline was tested using normal input, unseen categorical values, and missing values.



**## 6. Baseline Comparisons**



Simple baselines establish how much value the machine-learning models add over straightforward historical predictions.



**### 6.1 Held-out test results**



\| Baseline | MAE (kg/ha) | RMSE (kg/ha) | R² |

\|---|---:|---:|---:|

\| Overall median | 1,159.28 | 1,718.25 | -0.4272 |

\| Crop median | 953.73 | 1,514.16 | -0.1083 |

\| Crop-state median | 868.36 | 1,362.17 | 0.1031 |



The crop-state median baseline was the strongest of these three baselines.



The baseline predicts from historical group medians rather than learning the full relationship between the selected environmental, temporal, geographic, and crop features.



**### 6.2 Why retain these baselines?**



Baselines provide an interpretable reference. A complicated model is not useful merely because it is more sophisticated; it should demonstrate a meaningful improvement over a simple alternative.



The 300-tree Extra Trees model later achieved approximately 49.1% lower test MAE than the crop-state median baseline.



**## 7. Initial Algorithm Comparison**



The initial comparison evaluated multiple regression approaches using the same general feature configuration and the 2013–2014 validation period.



**### 7.1 Validation results**



\| Model | MAE | RMSE | R² |

\|---|---:|---:|---:|

\| Ridge Regression | 645.02 | 997.44 | 0.4680 |

\| Random Forest, 300 trees | 403.85 | 683.89 | 0.7499 |

\| Extra Trees, 300 trees | 387.73 | 669.49 | 0.7603 |

\| Gradient Boosting | 638.04 | 1,075.78 | 0.3811 |

\| HistGradientBoosting | 442.82 | 690.05 | 0.7454 |

\| XGBoost | 437.96 | 687.94 | 0.7469 |

\| LightGBM | 431.39 | 672.88 | 0.7579 |

\| Random Forest with log-transformed target | 404.51 | 691.27 | 0.7445 |

\| CatBoost | 428.76 | 686.14 | 0.7482 |



*\*All MAE and RMSE values are in kg/ha.\**



**### 7.2 Interpretation**



Extra Trees produced the best validation MAE, RMSE, and R² among the listed initial comparisons.



LightGBM was competitive, especially in R², and remained a useful alternative. The results also showed that model complexity alone does not guarantee stronger performance: Ridge Regression and Gradient Boosting had considerably higher validation errors in these experiments.



**### 7.3 Crop-level validation comparison**



\| Crop | Extra Trees MAE | Random Forest MAE | LightGBM MAE |

\|---|---:|---:|---:|

\| Chickpea | 294.50 | 292.20 | 284.11 |

\| Cotton | 114.51 | 116.15 | 165.66 |

\| Maize | 680.75 | 708.86 | 712.27 |

\| Rice | 315.84 | 344.20 | 418.14 |



Extra Trees was not the best model for every crop. LightGBM had lower chickpea MAE, while Extra Trees performed better on cotton, maize, and rice in this comparison.



This is why crop-level results should accompany overall metrics.



**## 8. Hyperparameter Tuning**



**### 8.1 Objective**



Hyperparameter tuning was used to investigate whether different model configurations could improve generalization.



Randomized search explored selected hyperparameter combinations. Cross-validation folds were chronological and confined to the training period, avoiding the use of the later validation and test years during tuning.



**### 8.2 Chronological cross-validation folds**



\| Fold | Training years | Validation years | Training rows | Validation rows |

\|---|---|---|---:|---:|

\| 1 | Through 2000 | 2001–2004 | 34,156 | 3,919 |

\| 2 | Through 2004 | 2005–2008 | 38,075 | 3,953 |

\| 3 | Through 2008 | 2009–2012 | 42,028 | 3,897 |



The tuning objective was to minimize MAE across these chronological validation folds.



**### 8.3 Best Extra Trees configuration**



The search selected:



\| Parameter | Value |

\|---|---|

\| \`n_estimators\` | 100 |

\| \`min_samples_leaf\` | 2 |

\| \`max_features\` | 1.0 |

\| \`max_depth\` | \`None\` |

\| Best cross-validation MAE | 305.81 kg/ha |



**### 8.4 Best LightGBM configuration**



The search selected:



\| Parameter | Value |

\|---|---|

\| \`reg_lambda\` | 0.0 |

\| \`num_leaves\` | 63 |

\| \`n_estimators\` | 600 |

\| \`min_child_samples\` | 10 |

\| \`learning_rate\` | 0.03 |

\| Best cross-validation MAE | 330.52 kg/ha |



These are the best configurations found within the defined search spaces, not proof of globally optimal hyperparameters.



**### 8.5 Tuned validation results**



\| Model | MAE | RMSE | R² |

\|---|---:|---:|---:|

\| Tuned Extra Trees | 387.21 | 670.24 | 0.7598 |

\| Tuned LightGBM | 409.88 | 667.60 | 0.7617 |



Tuned Extra Trees improved slightly over the initial Extra Trees validation result. Tuned LightGBM improved over the initial LightGBM MAE.



These differences were modest. Validation results alone were not used to decide which model should replace the existing Random Forest.



**## 9. Held-Out Test Evaluation**



The final comparison used the test years 2015–2017, containing 2,918 rows. These years were not used for model fitting or hyperparameter selection.



**### 9.1 Overall test results**



\| Model | MAE | RMSE | R² | Mean error |

\|---|---:|---:|---:|---:|

\| Extra Trees baseline, 300 trees | 441.89 | 810.44 | 0.6825 | -135.93 |

\| Tuned Extra Trees | 442.03 | 811.45 | 0.6817 | -135.35 |

\| Existing Random Forest | 454.09 | 821.56 | 0.6737 | -130.50 |

\| Tuned LightGBM | 457.22 | 820.90 | 0.6742 | -141.77 |

\| Crop-state median baseline | 868.36 | 1,362.17 | 0.1031 | -782.24 |



*\*All error values are in kg/ha.\**



**### 9.2 Main findings**



**\*\*Extra Trees with 300 trees\*\*** achieved the lowest overall test MAE among the listed models.



Compared with the existing Random Forest:



\- Test MAE decreased from 454.09 to 441.89 kg/ha, an improvement of approximately 2.69%.

\- Test RMSE decreased from 821.56 to 810.44 kg/ha.

\- Test R² increased from 0.6737 to 0.6825.



These results support Extra Trees as the leading candidate among the tested configurations. The improvement is real in the reported metrics but modest; it should not be overstated.



**### 9.3 Crop-level test MAE**



\| Crop | Crop-state median | Existing Random Forest | Extra Trees, 300 trees | Tuned Extra Trees | Tuned LightGBM |

\|---|---:|---:|---:|---:|---:|

\| Chickpea | 426.67 | 255.15 | 260.31 | 261.04 | 252.81 |

\| Cotton | 278.63 | 222.67 | 219.93 | 219.96 | 241.31 |

\| Maize | 1,490.82 | 803.67 | 777.92 | 778.49 | 801.15 |

\| Rice | 969.24 | 411.91 | 391.79 | 390.98 | 418.01 |



Extra Trees with 300 trees performed better than the existing Random Forest on cotton, maize, and rice, but not chickpea.



Tuned LightGBM achieved the lowest chickpea MAE in this comparison, while tuned Extra Trees achieved the lowest rice MAE. This demonstrates a trade-off between overall performance and performance on individual crops.



**### 9.4 Error bias**



The Extra Trees model's mean test error was approximately -135.93 kg/ha under the report's signed-error convention. This indicates average underprediction.



Average bias does not describe the full error distribution. Individual predictions can still be substantially wrong, particularly for unusual observations and extreme yields.



**## 10. Why Tuning Did Not Improve Test Performance**



Tuned Extra Trees achieved test MAE of 442.03 kg/ha, compared with 441.89 kg/ha for the original 300-tree Extra Trees model.



This does not mean hyperparameter tuning is useless. It means that the best configuration under one validation objective did not produce a better result on this later test period.



Potential explanations include:



1\. **\*\*Different optimization periods:\*\*** cross-validation and the test set represent different historical periods.

2\. **\*\*Temporal distribution shift:\*\*** relationships between crop yield, weather, and farming practices can change over time.

3\. **\*\*Limited search space:\*\*** randomized search evaluates selected configurations rather than every possible configuration.

4\. **\*\*Different ensemble sizes:\*\*** tuning selected 100 trees instead of the baseline's 300 trees.

5\. **\*\*Small performance difference:\*\*** the test MAE difference is only about 0.14 kg/ha, which is negligible relative to an MAE above 440 kg/ha.



The correct conclusion is that the tuned Extra Trees model did not demonstrate a meaningful improvement over the 300-tree Extra Trees model on this test set.



Hyperparameter tuning is a process for finding promising configurations, not a guarantee that every tuned model will outperform its baseline.



**## 11. Candidate Artifact Creation**



After comparing models and evaluating the held-out test set, a separate Extra Trees candidate pipeline was saved.



**\*\*Candidate artifact\*\***



\`models/crop_yield_extratrees_candidate.joblib\`



**\*\*Candidate metadata\*\***



\`models/crop_yield_extratrees_candidate_metadata.json\`



The candidate uses the Extra Trees configuration with:



\- \`n_estimators=300\`

\- \`min_samples_leaf=2\`

\- \`random_state=42\`

\- \`n_jobs=-1\`



It uses the selected categorical and numeric features and the pipeline preprocessing described earlier.



The candidate was saved separately from the existing Random Forest artifact:



\`models/crop_yield_pipeline.joblib\`



The existing artifact was not overwritten during candidate training or verification.



The test metrics recorded for the candidate were taken from the earlier held-out evaluation. Saving the candidate did not independently reproduce the full test evaluation. The metrics should be traced to the corresponding evaluation reports.



**## 12. Candidate Verification Results**



The verification script was executed with:



\`python training/verify_candidate_model.py\`



The reported checks were:



\| Verification | Result |

\|---|---|

\| Candidate artifact exists | Passed |

\| Candidate metadata exists | Passed |

\| Saved model loads | Passed |

\| Metadata model name is correct | Passed |

\| Metadata feature configuration matches | Passed |

\| Normal input prediction is finite | Passed |

\| Unseen categorical values are handled | Passed |

\| Missing input values are handled | Passed |

\| Existing Random Forest artifact remains present | Passed |



The sample predictions produced by the verification script were:



\| Test input | Prediction |

\|---|---:|

\| Normal input | 525.13 kg/ha |

\| Unseen categories | 650.15 kg/ha |

\| Missing values | 726.69 kg/ha |



These are smoke-test outputs for the selected test inputs, not measures of model accuracy.



Pandas emitted \`FutureWarning\` messages when the script assigned missing values to integer-inferred numeric columns. The script nevertheless completed successfully. The warnings should be cleaned up by making the test columns compatible with missing numeric values.



**### What verification establishes**



Verification establishes that the saved artifact can be loaded and produces finite predictions for the tested input scenarios.



It does not establish:



\- Accuracy on new real-world data.

\- Suitability for all Indian crops or regions.

\- Reliability for years far beyond the training period.

\- Correctness of every incoming feature value.

\- Successful end-to-end operation of the full Streamlit application.



Those questions require additional evaluation and integration testing.



**## 13. Model Selection Decision**



**\*\*Current leading candidate:\*\*** Extra Trees Regressor with 300 trees and \`min_samples_leaf=2\`.



Reasons:



1\. It achieved the best overall test MAE among the evaluated models.

2\. It also achieved the lowest test RMSE and highest R² among the listed models.

3\. It outperformed the existing Random Forest on overall held-out metrics.

4\. It substantially outperformed the crop-state median baseline.

5\. It was saved as a separate artifact and passed the candidate smoke tests.



The existing Random Forest remains the preserved model until the candidate passes end-to-end integration tests and the final replacement decision is made.



The candidate should not be promoted based only on the small metric improvement. Promotion must also account for compatibility with the application's input schema, output handling, artifact-loading code, and expected user-facing behavior.



**## 14. Known Limitations and Risks**



**### 14.1 Historical data**



The dataset ends in 2017. Current performance cannot be assumed from historical test metrics alone.



**### 14.2 Dataset provenance**



The original Kaggle listing could not be verified. The source and data-generation process remain uncertain.



**### 14.3 Potential leakage**



Nutrient variables showed suspiciously strong relationships with the target and were excluded. Their original calculation process remains unverified.



**### 14.4 Extreme observations**



Some crops contain unusually large target values. They were investigated, but they should not be removed without defensible source evidence and a documented policy.



**### 14.5 Temporal generalization**



The chronological test split is more realistic than a random split for forecasting future periods, but it still represents only 2015–2017. It does not prove performance under present-day conditions.



**### 14.6 Geographic and crop variation**



The model's overall MAE hides differences between crops. Per-crop evaluation should remain part of future monitoring and model reviews.



**### 14.7 Prediction is not agronomic advice**



A yield estimate is not a guarantee of harvest outcome. The AI Advisor must not present the model prediction as certainty or as a substitute for validated local agronomic recommendations.



**### 14.8 Feature availability**



Production use requires checking that every input is available, measured consistently, and provided in the same units as the training data. Missing values and unknown categories being accepted by the pipeline does not automatically make every input meaningful.



**## 15. Next Steps Before Production Promotion**



1\. Clean up the pandas warnings in the candidate verification script.

2\. Review the candidate metadata and evaluation report paths for consistency.

3\. Confirm the application loads the candidate using the expected feature schema.

4\. Run targeted integration tests for prediction inputs, output units, error handling, and application behavior.

5\. Confirm that the original Random Forest artifact remains intact and that rollback is possible.

6\. Compare candidate and current-model outputs on representative application inputs.

7\. Document the final promotion decision and the reason for it.

8\. Consider further evaluation using newer, independently sourced agricultural data before relying on the model for real-world recommendations.



**## 16. Interview Preparation Notes**



**\*\*Why did you use a chronological split?\*\***



Because the intended task involves predicting yields for later periods. Training on earlier years and testing on later years better simulates future use and reduces the risk of overly optimistic results from random splitting.



**\*\*Why exclude the nutrient features?\*\***



Their correlations with the target were suspiciously strong, and their calculation process was unknown. Excluding them was a conservative way to reduce potential target leakage.



**\*\*Why use Extra Trees instead of Random Forest?\*\***



Extra Trees achieved lower held-out MAE and RMSE and a higher R² in our reported test comparison. The improvement was modest, so I treated it as a candidate requiring further integration validation rather than assuming it was automatically production-ready.



**\*\*Why did tuning not improve Extra Trees?\*\***



The tuning process optimized chronological cross-validation MAE within the training period. The selected parameters did not generalize better to the later test period. The observed difference was extremely small, showing why final selection must use held-out evaluation.



**\*\*Why preserve the existing Random Forest?\*\***



A candidate can perform well in offline evaluation but still fail during application integration. Keeping the original artifact untouched preserves a working fallback while the candidate is validated.



**\*\*What are the biggest limitations?\*\***



The dataset is historical and ends in 2017, its original provenance is unverified, some features raised potential leakage concerns, and model performance varies by crop. Additional contemporary data and real-world validation are needed before making strong production claims.



**---**



**## 17. Experiment Artifacts and Reports**



The following files contain supporting experiment outputs:



\- \`reports/model_comparison/validation_model_comparison.csv\`

\- \`reports/model_comparison/validation_crop_comparison.csv\`

\- \`reports/model_comparison/tuning_cv_results.csv\`

\- \`reports/model_comparison/tuned_validation_comparison.csv\`

\- \`reports/model_comparison/tuned_validation_crop_metrics.csv\`

\- \`reports/model_comparison/tuning_best_params.json\`

\- \`reports/model_comparison/test_model_comparison.csv\`

\- \`reports/model_comparison/test_crop_comparison.csv\`

\- \`reports/model_comparison/test_model_predictions.csv\`



The candidate files are:



\- \`models/crop_yield_extratrees_candidate.joblib\`

\- \`models/crop_yield_extratrees_candidate_metadata.json\`



The existing Random Forest artifact remains:



\- \`models/crop_yield_pipeline.joblib\`



**## Final Status**



**\*\*Candidate saved:\*\*** Yes  

**\*\*Candidate smoke verification:\*\*** Passed  

**\*\*Existing Random Forest preserved:\*\*** Yes  

**\*\*Candidate production promotion:\*\*** Pending end-to-end integration checks



This log should be updated whenever additional experiments are performed or the production model decision changes.


## 18. Modular Application Refactor Progress

This section records implementation and validation work completed after the model-comparison experiments above. It supplements the model-selection history; it does not replace or change the reported experiment results.

### 18.1 Refactor principles

The application is being reorganized into modules so that configuration, logging, location lookup, weather retrieval, soil-pH interpretation, yield prediction, and AI advice can be validated independently before being integrated into the Streamlit interface.

The following safeguards remain in effect:

- The original project under `C:\sih\training` is preserved and is not to be modified as part of the refactor.
- `planning1.2.md` is the source of truth for the implementation roadmap.
- The existing Random Forest pipeline remains the default model.
- Extra Trees remains an experimental candidate. It must not be promoted solely because of its offline metrics; end-to-end integration checks and a documented decision are still required.
- Existing model artifacts must not be deleted or overwritten before a replacement passes evaluation and integration checks.
- API keys and other secrets must remain in the project `.env` file and must not be printed, logged, or included in documentation.
- Package-style imports are used from the project root, for example `from kishan_mitra.config import ...`.

### 18.2 Current modular layout

The refactored project is under `C:\sih\kishan_mitra`. The following structure has been established or used during the work to date:

```text
C:\sih\kishan_mitra\
├── app.py
├── .env
├── requirements.txt
├── pyproject.toml
├── src\kishan_mitra\
│   ├── __init__.py
│   ├── config.py
│   ├── logging_config.py
│   ├── geocoding.py
│   ├── weather.py
│   ├── location_weather.py
│   ├── soil.py
│   ├── prediction.py
│   └── advisor.py
├── Datasets\Custom_Crops_yield_Historical_Dataset.csv
├── models\
├── training\
├── tests\
├── logs\
└── ...
```

This is a working progress snapshot, not a claim that every listed directory or module is complete. The project structure may evolve in line with `planning1.2.md`.

### 18.3 Configuration and dataset validation

The configuration module defines the project root relative to the package location and centralizes the dataset, model, and training paths. The configured dataset path is:

`Datasets/Custom_Crops_yield_Historical_Dataset.csv`

The dataset validation command succeeded when run from `C:\sih` using package/module execution:

```cmd
python -m kishan_mitra.data.validation
```

The successful validation report recorded 50,765 rows and 20 columns. The columns reported were:

- `Dist Code`
- `Year`
- `State Code`
- `State Name`
- `Dist Name`
- `Crop`
- `Area_ha`
- `Yield_kg_per_ha`
- `N_req_kg_per_ha`
- `P_req_kg_per_ha`
- `K_req_kg_per_ha`
- `Total_N_kg`
- `Total_P_kg`
- `Total_K_kg`
- `Temperature_C`
- `Humidity_%`
- `pH`
- `Rainfall_mm`
- `Wind_Speed_m_s`
- `Solar_Radiation_MJ_m2_day`

An earlier attempt to execute the validation file directly by its file path, and an attempt to run the module from inside the package directory, failed because the package import root was incorrect for those invocation locations. Running the module from `C:\sih` succeeded. Future package-module commands should be run from the project root appropriate to the package name.

### 18.4 Logging module

The logging setup was tested successfully. It writes application logs to `logs/kishan_mitra.log` and uses rotation with a 2 MB maximum file size and three backups. Repeated startup log entries can occur when Streamlit reruns the app; these entries alone do not establish a failure.

### 18.5 Geocoding and weather modules

The geocoding module was tested with Patna, Bihar and returned coordinates. The weather module was tested successfully against the configured weather service and returns the normalized fields:

- `temperature_c`
- `humidity_pct`
- `wind_speed_m_s`
- `precipitation_mm`

Wind speed is converted from kilometres per hour to metres per second. The weather module validates expected response fields and handles failures without logging the API key. The combined location-and-weather module was also tested successfully.

These tests establish that the tested calls worked at the time of testing; they do not guarantee uninterrupted availability of the external geocoding or weather services.

### 18.6 Soil pH interpretation module

The soil module's `analyze_soil_ph()` function was checked with a valid pH value and an out-of-range value. A pH of 6.5 was accepted and categorized as near neutral under the project's current thresholds. A pH value of 15 raised the expected `ValueError` because it is outside the accepted 0–14 range.

The module explicitly does not claim that pH alone determines soil fertility. Its output should remain a cautious interpretation rather than a complete soil-health assessment.

### 18.7 Prediction module and model-selection safeguards

The prediction module supports these model identifiers:

- `random_forest`, using `models/crop_yield_pipeline.joblib`
- `extra_trees_candidate`, using `models/crop_yield_extratrees_candidate.joblib`

The configured input schema is the three categorical features `State Name`, `Dist Name`, and `Crop`, together with the numeric features `Year`, `Temperature_C`, `Humidity_%`, `pH`, `Rainfall_mm`, `Wind_Speed_m_s`, and `Solar_Radiation_MJ_m2_day`.

Area, soil type, irrigation method, and previous crop are not inputs to the current yield-prediction pipeline. If collected by the user interface, these values must not be presented as though the current ML model directly used them. They may be passed as separate context to the AI Advisor where appropriate.

The historical dataset ends in 2017. A prediction requested for a year such as 2026 is an extrapolation beyond the observed period and must be labelled as such; historical test metrics do not validate future-year performance.

The application currently defaults to Random Forest and presents Extra Trees as an experimental candidate with a warning. This is intentional and must remain in place until the candidate's integration and replacement decision are completed.

## 19. AI Advisor Module — Implementation and Verification Log

### 19.1 Purpose and model configuration

The AI Advisor module is located at:

`src/kishan_mitra/advisor.py`

Its purpose is to generate concise, farmer-friendly guidance using the crop, location, predicted yield, soil pH, weather measurements, and optional context such as soil type, irrigation method, previous crop, and growth stage. The advisor is separate from the numeric yield-prediction model: its generated text must not be treated as a verified yield forecast or as a substitute for locally validated agronomic advice.

The configured Groq model identifier is `openai/gpt-oss-120b`. The model configuration was deliberately left unchanged during the latest verification. The call currently uses `temperature=0.3` and `max_tokens=700`.

The module loads environment variables from the project-root `.env` file using `python-dotenv`. The API key is read at runtime and must never be printed or included in logs or this document.

### 19.2 Function interface

The current public function interface is:

```python
def get_advisor_response(
    crop,
    state,
    district,
    predicted_yield_kg_per_ha,
    soil_ph,
    temperature_c,
    humidity_pct,
    rainfall_mm,
    wind_speed_m_s,
    solar_radiation_mj_m2_day,
    soil_type=None,
    irrigation=None,
    previous_crop=None,
    growth_stage=None,
):
    ...
```

The first ten arguments provide the required prediction and environmental context. Soil type, irrigation, previous crop, and growth stage are optional and must be treated as unknown when not supplied; they must not be silently assumed.

### 19.3 Safety and quality rules in the advisor prompt

The current system prompt instructs the advisor to:

1. Describe the yield as an estimate, not a guarantee.
2. Avoid promising a fixed percentage increase in yield.
3. Avoid inventing fertilizer or pesticide doses, prices, or return-on-investment figures.
4. Avoid prescribing a pesticide without an identified pest or disease.
5. Treat missing information as unknown, not as zero or as a known condition.
6. Avoid numeric irrigation depths, watering schedules, or fertilizer rates unless the crop stage, cultivation method, relevant conditions, and sound agronomic basis are available. Otherwise, explain the general principle and ask for the missing information.
7. Recommend soil testing or a local Krishi Vigyan Kendra (KVK)/agriculture officer when precise local quantities are needed.
8. Consider growth stage, and acknowledge when it has not been provided.
9. Avoid claiming live weather, market prices, or external verification unless that information has actually been supplied or verified.
10. Avoid inferring the growing season, planting date, or crop stage from location or weather values alone. Do not call conditions typical for a region or season without supplied or verified support; explain that interpretation depends on crop stage and local conditions.

The intended response structure is a brief assessment, three practical actions, one important caution, and a request for additional information where relevant. Responses should remain concise and understandable to farmers.

### 19.4 Tests performed and results

The following checks have been performed on the current advisor implementation:

| Check | Method or scenario | Result |
|---|---|---|
| Module import | Import `get_advisor_response` through `kishan_mitra.advisor` | Passed |
| Live response | Invoke the configured model with a complete sample context | Passed; a non-empty response was returned |
| Response completion diagnostics | Inspect response finish reason and usage metadata during a temporary diagnostic run | The tested response had finish reason `stop`; the reported output usage was 651 tokens |
| Missing optional context | Invoke with the required fields and omit soil type, irrigation, previous crop, and growth stage | Passed; the response acknowledged missing context and asked for relevant information |
| Missing API key | Mock environment lookup to return no key | Passed; raised the expected `RuntimeError` with a message directing the developer to check the project `.env` file |
| Simulated API failure | Mock the model invocation to raise a runtime error | Passed; the error was logged and re-raised rather than silently swallowed |
| Empty/non-text response guard | Mock an all-whitespace response body | Passed; raised the expected `RuntimeError` for an empty or non-text advisor response |
| Temporary debug statement cleanup | Search for `DEBUG_FINISH_REASON`, `DEBUG_USAGE`, `max_tokens`, and `finish_reason` | Passed; only the intended `max_tokens` and `finish_reason` handling remained in the search output |
| Final import check | Run `python -c "from kishan_mitra.advisor import get_advisor_response; print('Advisor import successful.')"` from the project root | Passed; user confirmed completion |

The live-response tests used sample inputs, including rice in Patna, Bihar and a sample predicted yield of 2,500 kg/ha. That number was a test value, not a claim about actual local yield or a verified prediction. The response checks establish that the function worked for those tested cases; they do not establish agricultural correctness for every recommendation or guarantee future API availability.

### 19.5 Logging and error handling

The advisor logs successful completion with crop and location context. On exceptions, it logs the failure and re-raises the exception so the caller can handle the failure explicitly. The API key is not intended to be logged.

The response is checked to ensure its content is a non-empty string. The code also checks `response.response_metadata.get("finish_reason")`; if the finish reason is `length`, it logs a warning that the response may be truncated.

Temporary `print()` diagnostics for finish reason and token usage were used during investigation and then removed. The latest source search confirmed that the debug identifiers `DEBUG_FINISH_REASON` and `DEBUG_USAGE` are absent while the intended finish-reason handling remains. `max_tokens=700` was retained because the diagnostic response completed with finish reason `stop`; no evidence from that test justified changing the configured limit.

### 19.6 Current advisor status and remaining work

**Status: Module-level checks completed for the scenarios listed above.** The final import check has been confirmed by the user.

The advisor has not yet been declared fully integrated with the Streamlit application. Remaining work includes connecting the module to the UI at the appropriate point in the roadmap, validating input handling and user-facing errors in the integrated application, and checking that advice is displayed with the necessary limitations and without implying that optional information was known.

No claim is made here that the advisor's agricultural recommendations have been independently reviewed by an agronomist. It is an AI-generated guidance feature and should be presented accordingly.

## 20. Consolidated Current Status and Next Actions

This section records the current state after the modular checks and advisor tests. It supplements the earlier model-experiment status rather than changing the held-out evaluation results.

### Completed or checked to date

- The historical dataset validation command completed successfully from the correct package root and reported 50,765 rows and 20 columns.
- The configuration, logging, geocoding, weather, combined location/weather, soil-pH, and prediction modules have been developed or checked as described in Section 18.
- The advisor import, live response, missing optional information, missing API key, simulated API failure, empty-response guard, and temporary-debug cleanup checks passed as recorded in Section 19.
- The Random Forest artifact remains the default and preserved model; Extra Trees remains a candidate rather than an approved production replacement.
- The AI Advisor's configured model and `max_tokens=700` setting remain unchanged.

### Next actions

1. Continue with the next module or integration step specified by `planning1.2.md`, validating only meaningful behavior and waiting for the actual command/test output before recording a test as passed.
2. Complete any remaining prediction-module tests and confirm the input schema, output units, and error behavior match the app's expectations.
3. Integrate the advisor and other ready modules into the Streamlit app only when the roadmap calls for it; then run focused end-to-end tests.
4. Keep the current Random Forest as the default until candidate integration, rollback readiness, and the final model decision are documented.
5. Revisit the pending candidate verification-script pandas warnings and confirm report/metadata paths, as listed in Section 15, if those items have not already been completed.
6. Update this log after each meaningful experiment or validation milestone. Record actual commands, observed outputs, and limitations; do not mark unrun checks as passed.

### Current decision snapshot

| Item | Current status |
|---|---|
| Existing Random Forest artifact | Preserved; remains the default |
| Extra Trees candidate artifact | Saved and smoke-verified according to the existing experiment record |
| Extra Trees production promotion | Pending end-to-end integration and final review |
| Dataset provenance | Unverified, as documented earlier |
| Historical dataset end year | 2017 |
| Modular refactor | In progress |
| Advisor module checks listed in Section 19 | Passed |
| Full Streamlit end-to-end validation | Pending |
| Agronomic validation by a qualified local expert | Not established by these software tests |

---

**Maintenance rule:** Keep this document as the cumulative record. Preserve earlier experiment results, add dated or clearly ordered entries for new work, distinguish offline model evaluation from software smoke tests, and explicitly label unknown or unverified information. Do not infer success for a test that has not actually been run.
