# 🌾 Kishan Mitra — Crop Yield Prediction & AI Advisor

Kishan Mitra is a machine-learning-powered agricultural decision-support application that estimates crop yield and provides AI-generated recommendations using crop, location, soil pH, and available weather information.

The project combines a scikit-learn regression pipeline with weather and geocoding services, soil pH analysis, and an AI Advisor. The application is built with Streamlit and is deployed on Streamlit Community Cloud.

**Live application:** [Kishan Mitra](https://kishanmitra-gxuhizi4wxd57sqqujrmfe.streamlit.app/)

> **Important:** Predicted yields are estimates, not guarantees of actual harvest. AI-generated recommendations should be reviewed against local agricultural guidance.

## Features

- **Crop-yield prediction:** Estimates yield in kilograms per hectare (kg/ha) using a saved machine-learning pipeline.
- **Modular ML architecture:** Separates prediction logic, configuration, data validation, and external services.
- **Weather integration:** Retrieves current weather information through WeatherAPI.
- **Location geocoding:** Converts supported locations into coordinates using OpenStreetMap Nominatim.
- **Soil pH analysis:** Classifies the supplied soil pH into a descriptive category.
- **AI Advisor:** Generates crop-specific guidance using an LLM service through Groq.
- **Model experimentation:** Includes scripts for baseline comparison, model evaluation, hyperparameter tuning, leakage checks, and candidate verification.
- **Logging and testing:** Records application events and errors and includes automated prediction tests.

## Technology Stack

| Component | Technology |
|---|---|
| User interface | Streamlit |
| Language | Python |
| Machine learning | scikit-learn |
| Data processing | pandas, NumPy |
| Model serialization | Joblib |
| Weather service | WeatherAPI |
| Geocoding | OpenStreetMap Nominatim |
| AI Advisor | LangChain and Groq |
| Testing | pytest |
| Packaging | `pyproject.toml` |
| Deployment | Streamlit Community Cloud |

## Project Structure

```text
kishan_mitra/
├── app.py
├── pyproject.toml
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── Datasets/
│   └── Custom_Crops_yield_Historical_Dataset.csv
├── data/
│   ├── __init__.py
│   ├── README.md
│   └── validation.py
├── docs/
│   ├── architecture.md
│   └── model-card.md
├── experiments/
├── logs/
├── models/
│   ├── crop_yield_pipeline.joblib
│   ├── model_metadata.json
│   ├── crop_yield_extratrees_candidate.joblib
│   └── crop_yield_extratrees_candidate_metadata.json
├── reports/
│   └── model_comparison/
├── src/
│   └── kishan_mitra/
│       ├── __init__.py
│       ├── config.py
│       ├── prediction.py
│       ├── weather.py
│       ├── geocoding.py
│       ├── location_weather.py
│       ├── soil.py
│       ├── advisor.py
│       └── logging_config.py
├── tests/
│   └── test_prediction.py
└── training/
    ├── README.md
    ├── model_config.py
    ├── train_model.py
    ├── train_candidate_model.py
    ├── compare_baselines.py
    ├── compare_models.py
    ├── evaluate_model.py
    ├── evaluate_tuned_models.py
    ├── tune_models.py
    ├── audit_dataset.py
    ├── check_feature_leakage.py
    ├── investigate_extreme_yields.py
    └── verify_candidate_model.py
```

The exact contents of optional folders such as `experiments/` may evolve as the project develops.

## Architecture

The application uses a modular structure:

1. `app.py` provides the Streamlit interface and coordinates the user workflow.
2. `src/kishan_mitra/prediction.py` loads the saved pipeline and generates yield estimates.
3. `weather.py`, `geocoding.py`, and `location_weather.py` support location and weather retrieval.
4. `soil.py` handles soil pH analysis.
5. `advisor.py` prepares AI Advisor requests and processes generated recommendations.
6. `config.py` centralizes project paths and model configuration.
7. `training/` contains the scripts used to audit data, train models, compare alternatives, and evaluate results.

See [docs/architecture.md](docs/architecture.md) for the detailed architecture and data flow.

## Dataset

The project uses `Datasets/Custom_Crops_yield_Historical_Dataset.csv`.

The dataset validation report recorded **50,765 rows and 20 columns**. It contains historical agricultural information, including crop, district, state, year, yield, nutrient-related fields, and environmental variables.

The target variable is:

`Yield_kg_per_ha`

The model's preprocessing pipeline and selected feature columns are stored with the serialized estimator. The application should use the same preprocessing logic as the training pipeline.

Dataset provenance, coverage, potential bias, and feature limitations should be reviewed before using predictions for real-world agricultural decisions.

## Machine-Learning Evaluation

The project evaluates candidate regressors using regression metrics, including:

- **MAE:** Mean Absolute Error
- **RMSE:** Root Mean Squared Error
- **R²:** Coefficient of Determination

The documented held-out test comparison used data from **2015–2017 (2,918 rows)** that was not used for model fitting or hyperparameter selection in that comparison.

| Model | Test MAE (kg/ha) | Test RMSE (kg/ha) | Test R² |
|---|---:|---:|---:|
| Existing Random Forest | 454.09 | 821.56 | 0.6737 |
| Extra Trees, 300 trees | 441.89 | 810.44 | 0.6825 |
| Tuned Extra Trees | 442.03 | 811.45 | 0.6817 |
| Tuned LightGBM | 457.22 | 820.90 | 0.6742 |

These are recorded experimental results, not a guarantee of future performance. The Extra Trees model achieved the lowest overall test MAE in this comparison, but its improvement over Random Forest was modest and varied by crop.

**Deployment status:** The application is configured to use Random Forest. Extra Trees remains an experimental candidate until its artifact, dependencies, prediction interface, and end-to-end deployment are verified together.

For the complete evaluation details and limitations, see [docs/model-card.md](docs/model-card.md) and `reports/model_comparison/`.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ayush16082006/KishanMitra.git
cd KishanMitra
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation is restricted by your PowerShell policy, use the Python executable inside `.venv` directly.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Use the dependency versions specified by the project. The saved model was created with scikit-learn 1.7.2, so maintain the compatible version specified in the requirements file.

### 4. Configure API credentials

Create a local `.env` file from `.env.example` and fill in your own credentials:

```powershell
Copy-Item .env.example .env
```

Configure the variables required by the current application:

```dotenv
WEATHER_API_KEY=your_weatherapi_key
GROQ_API_KEY=your_groq_api_key
```

Never commit `.env` or expose API credentials in logs, screenshots, source code, or public documentation.

For Streamlit Community Cloud, add the corresponding values in the app's **Settings → Secrets** section instead of committing them.

### 5. Validate the dataset

From the project root:

```powershell
python -m kishan_mitra.data.validation
```

This command assumes the package is installed in the active environment and the project's import paths are configured as expected.

### 6. Run the application

```powershell
streamlit run app.py
```

Streamlit will display a local URL in the terminal.

## Testing

Run the available automated tests from the project root:

```powershell
python -m pytest tests/ -v
```

The current automated tests include prediction-related checks. External API behavior and the complete deployed workflow should also be verified separately.

Before releasing a change, check that:

- The saved model loads successfully.
- A valid input produces a finite, non-negative prediction.
- The prediction uses the intended model.
- Weather or geocoding failures are handled clearly.
- AI Advisor failures do not misrepresent a prediction.
- API credentials remain private.

## Training and Evaluation

Training scripts are maintained separately from the application code under `training/`.

Examples:

```powershell
python -m training.audit_dataset
python -m training.compare_models
python -m training.evaluate_model
python -m training.tune_models
```

Check the relevant script and `training/README.md` before running any command. Some scripts can take significant time or memory, and training is not required to run the deployed application.

Model-comparison reports are stored under `reports/model_comparison/`.

Do not replace the production model simply because a candidate has a better score in one metric. Consider the held-out test results, performance by crop, reproducibility, artifact compatibility, application integration, and resource requirements.

## Environment Variables and Secrets

| Variable | Purpose |
|---|---|
| `WEATHER_API_KEY` | Access to WeatherAPI |
| `GROQ_API_KEY` | Access to the Groq-powered AI Advisor |

Additional configuration may be required if the integrations change. Check `.env.example`, `config.py`, `weather.py`, and `advisor.py` for the current implementation.

## Limitations

- Model predictions can be inaccurate for unusual locations, crops, years, or environmental conditions.
- Aggregate evaluation metrics do not describe the error of every individual prediction.
- Performance differs across crops.
- Historical data may not represent current farming practices, climate, soil conditions, or future growing seasons.
- Current weather is not equivalent to a full seasonal forecast.
- Geocoding and weather depend on external services, which may be unavailable or rate-limited.
- AI-generated advice may be incomplete or incorrect and should not replace local agricultural expertise.
- The current model's input features and data coverage constrain how confidently predictions can be generalized.
- Model artifacts can require substantial memory and compatible dependency versions.

## Responsible Use

Use Kishan Mitra as an informational decision-support tool. Verify important decisions—especially fertilizer quantities, pesticide use, irrigation changes, and major financial commitments—with qualified local agricultural professionals and authoritative local recommendations.

The application does not guarantee a specific yield increase, profit, or harvest outcome.

## Documentation

- [Architecture](docs/architecture.md)
- [Model Card](docs/model-card.md)
- [Training Guide](training/README.md)
- [Dataset Validation](data/README.md)

## License

No license is specified here. Add a license file if you intend to grant others permission to use, modify, or distribute this project.
