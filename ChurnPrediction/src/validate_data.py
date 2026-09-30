import os
import json
import pandas as pd
import numpy as np

def run_data_validation():
    print("[INFO] Starting Raw Data Validation...")
    
    data_path = 'data/raw/churn.csv'
    if not os.path.exists(data_path):
        print(f"[ERROR] Data file not found at {data_path}")
        return False
        
    df = pd.read_csv(data_path)
    errors = []
    
    # 1. Check Row Count
    if len(df) == 0:
        errors.append("Dataset is empty.")
        
    # 2. Check Expected Columns
    required_columns = [
        'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
    ]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        
    # 3. Check Value Constraints
    if 'SeniorCitizen' in df.columns:
        invalid_sc = df[~df['SeniorCitizen'].isin([0, 1])]
        if len(invalid_sc) > 0:
            errors.append(f"SeniorCitizen contains invalid values: {len(invalid_sc)} rows.")
            
    if 'tenure' in df.columns:
        invalid_tenure = df[df['tenure'] < 0]
        if len(invalid_tenure) > 0:
            errors.append(f"Negative tenure values found: {len(invalid_tenure)} rows.")
            
    if 'MonthlyCharges' in df.columns:
        invalid_monthly = df[df['MonthlyCharges'] < 0]
        if len(invalid_monthly) > 0:
            errors.append(f"Negative MonthlyCharges found: {len(invalid_monthly)} rows.")
            
    if 'Churn' in df.columns:
        valid_churn = {'Yes', 'No', 'yes', 'no', '1', '0', 1, 0}
        invalid_churn = df[~df['Churn'].isin(valid_churn)]
        if len(invalid_churn) > 0:
            errors.append(f"Unexpected values in Churn column: {len(invalid_churn)} rows.")
            
    # 4. Generate Validation Report
    report = {
        "test_name": "Raw Data Validation",
        "validation_status": "PASSED" if not errors else "FAILED",
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "columns_checked": required_columns,
        "errors": errors
    }
    
    os.makedirs("artifacts", exist_ok=True)
    report_path = 'artifacts/data_validation_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    if errors:
        print("[ERROR] Data Validation FAILED:")
        for err in errors:
            print(f"  - {err}")
        return False
    else:
        print(f"[SUCCESS] Data Validation PASSED! Verified {len(df)} records.")
        print(f"[INFO] Report saved to {report_path}")
        return True

if __name__ == "__main__":
    success = run_data_validation()
    if not success:
        exit(1)
