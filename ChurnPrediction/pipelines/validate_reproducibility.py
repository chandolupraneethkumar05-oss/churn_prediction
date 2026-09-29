import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_PATH = PROJECT_ROOT / "artifacts" / "reproducibility_report.json"


def run_deterministic_test():
    print("Running Reproducibility Validation...")

    # Load Data
    X_train = np.load(PROCESSED_DATA_DIR / 'X_train_final.npy')
    X_test = np.load(PROCESSED_DATA_DIR / 'X_test_final.npy')
    y_train = np.load(PROCESSED_DATA_DIR / 'y_train.npy')
    y_test = np.load(PROCESSED_DATA_DIR / 'y_test.npy')

    # Fixed Parameters for strict reproducibility
    params = {"n_estimators": 100, "max_depth": 10, "random_state": 42, "class_weight": "balanced"}

    # Execution 1
    model_1 = RandomForestClassifier(**params)
    model_1.fit(X_train, y_train)
    score_1 = f1_score(y_test, model_1.predict(X_test))

    # Execution 2 (Simulating a re-run)
    model_2 = RandomForestClassifier(**params)
    model_2.fit(X_train, y_train)
    score_2 = f1_score(y_test, model_2.predict(X_test))

    # Validate consistency
    is_reproducible = (score_1 == score_2)

    # Generate Report
    report = {
        "test_name": "Pipeline Reproducibility Validation",
        "parameters": params,
        "execution_1_f1": score_1,
        "execution_2_f1": score_2,
        "is_strictly_reproducible": is_reproducible,
        "status": "PASSED" if is_reproducible else "FAILED"
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=4), encoding='utf-8')

    print(f"Execution 1 F1: {score_1:.6f}")
    print(f"Execution 2 F1: {score_2:.6f}")
    if is_reproducible:
        print("SUCCESS: Pipeline is 100% reproducible. Report saved.")
    else:
        raise RuntimeError("Pipeline is non-deterministic.")

if __name__ == "__main__":
    run_deterministic_test()
