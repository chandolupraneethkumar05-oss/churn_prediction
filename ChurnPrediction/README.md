# Telco Customer Churn Prediction

This project preprocesses the Telco customer dataset, trains a random-forest baseline, and evaluates its predictions.

## Project layout

- `data/raw/` contains the source CSV.
- `data/processed/` contains the checked-in tabular datasets and generated feature arrays.
- `src/` contains the preprocessing, training, and evaluation scripts.
- `models/` and `outputs/` are created as needed for generated models and error-analysis CSVs.
- `notebooks/` contains the exploratory and model-comparison notebook.

## Setup and run

From this directory, install the project dependencies and run each step in order:

```powershell
python -m pip install -r requirements.txt
python src/preprocess.py
python src/train.py
python src/evaluate.py
```

The scripts resolve data and output paths relative to their own location, so they can also be launched from another working directory.

To run the notebook, open `notebooks/project_implementation.ipynb` with Jupyter and run its cells in order.
