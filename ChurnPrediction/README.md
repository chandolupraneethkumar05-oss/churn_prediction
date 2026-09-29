# Telco Customer Churn Prediction

This project includes a baseline churn model, MLflow experiment tracking, a preprocessing pipeline, reproducibility/output checks, and an MLflow model registry lifecycle.

## Project layout

- `data/raw/` contains the source CSV.
- `data/processed/` contains the checked-in tabular datasets and generated feature arrays.
- `src/` contains the preprocessing, training, and evaluation scripts.
- `pipelines/` contains runnable Lab 3, Lab 4, and Lab 5 orchestrators, the preprocessing pipeline, and reproducibility check.
- `models/` contains model artifacts and the Lab 6 model-registry runner.
- `outputs/` contains output validation.
- `reports/` contains the model-registry report generator.
- `artifacts/` and `mlflow.db` are generated when the validation and registry workflows run.
- `notebooks/` contains the exploratory and model-comparison notebook.

## Setup and run

From this directory, install the project dependencies and run each step in order:

```powershell
python -m pip install -r requirements.txt
python src/preprocess.py
python src/train.py
python src/evaluate.py
```

Run commands from this directory. The main scripts resolve paths relative to their own location; the pipeline runners set the project directory before launching their child scripts.

To run the baseline workflow:

```powershell
python pipelines/run_lab3_baseline.py
```

To run MLflow experiment tracking and reproducibility validation:

```powershell
python pipelines/run_lab4_tracking.py
```

To run the modular preprocessing pipeline and validate its outputs:

```powershell
python pipelines/run_lab5_pipeline.py
```

After Lab 5 has created `models/preprocessor.pkl`, train and promote a model through the registry:

```powershell
python models/run_lab6_registry.py
```

The Lab 6 workflow requires the local MLflow registry database and model artifacts it creates. Do not run it against an MLflow tracking store containing production models unless you intend to change model stages.

To run the notebook, open `notebooks/project_implementation.ipynb` with Jupyter and run its cells in order.
