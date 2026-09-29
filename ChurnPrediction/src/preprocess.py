import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn (1).csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def run_preprocessing():
    print("Starting Preprocessing Pipeline...")

    # 1. Load the raw data
    df = pd.read_csv(RAW_DATA_PATH)

    # 2. Data Cleaning
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    target = df['Churn'].astype('string').str.strip().str.casefold()
    invalid_target = ~target.isin(['yes', 'no'])
    if invalid_target.any():
        invalid_values = target[invalid_target].unique().tolist()
        raise ValueError(f"Unexpected values in Churn column: {invalid_values}")
    df['Churn'] = target.map({'yes': 1, 'no': 0}).astype(int)

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Separate Column Types
    cat_cols = X_train.select_dtypes(include=['object', 'category', 'string']).columns
    num_cols = X_train.select_dtypes(include=['number']).columns

    # 5. Scale & Encode
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(X_train[num_cols])
    x_test_scaled = scaler.transform(X_test[num_cols])

    ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    x_train_encoded = ohe.fit_transform(X_train[cat_cols])
    x_test_encoded = ohe.transform(X_test[cat_cols])

    # Combine Features
    X_train_final = np.hstack((x_train_scaled, x_train_encoded))
    X_test_final = np.hstack((x_test_scaled, x_test_encoded))

    # 6. Save artifacts
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DATA_DIR / 'X_train_final.npy', X_train_final)
    np.save(PROCESSED_DATA_DIR / 'X_test_final.npy', X_test_final)
    np.save(PROCESSED_DATA_DIR / 'y_train.npy', y_train.to_numpy(dtype=np.int64))
    np.save(PROCESSED_DATA_DIR / 'y_test.npy', y_test.to_numpy(dtype=np.int64))

    joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')
    joblib.dump(ohe, MODELS_DIR / 'ohe.pkl')

    # Save Metadata
    metadata = {
        "dataset_name": "Telco Customer Churn",
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "numerical_features": list(num_cols),
        "categorical_features": list(cat_cols)
    }
    with (PROCESSED_DATA_DIR / 'dataset_metadata.json').open('w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)

    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    run_preprocessing()