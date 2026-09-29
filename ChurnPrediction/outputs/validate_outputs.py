import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_PATH = PROJECT_ROOT / "artifacts" / "preprocessing_summary_report.json"


def validate_preprocessing_outputs():
    print("[INFO] Validating preprocessing outputs...")

    try:
        X_train = np.load(PROCESSED_DATA_DIR / "X_train_final.npy")
        X_test = np.load(PROCESSED_DATA_DIR / "X_test_final.npy")
        y_train = np.load(PROCESSED_DATA_DIR / "y_train.npy")
        y_test = np.load(PROCESSED_DATA_DIR / "y_test.npy")
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Could not find processed data: {error.filename}"
        ) from error

    errors = []
    if np.isnan(X_train).any():
        errors.append("NaNs detected in X_train after preprocessing.")
    if np.isnan(X_test).any():
        errors.append("NaNs detected in X_test after preprocessing.")
    if X_train.ndim != 2 or X_test.ndim != 2:
        errors.append("Feature arrays must be two-dimensional.")
    elif X_train.shape[1] != X_test.shape[1]:
        errors.append(
            f"Feature mismatch: X_train has {X_train.shape[1]} columns, "
            f"X_test has {X_test.shape[1]} columns."
        )
    if X_train.shape[0] != y_train.shape[0]:
        errors.append("Row mismatch between X_train and y_train.")
    if X_test.shape[0] != y_test.shape[0]:
        errors.append("Row mismatch between X_test and y_test.")

    report = {
        "validation_status": "PASSED" if not errors else "FAILED",
        "matrix_dimensions": {
            "X_train_shape": list(X_train.shape),
            "X_test_shape": list(X_test.shape),
            "y_train_shape": list(y_train.shape),
            "y_test_shape": list(y_test.shape),
        },
        "data_quality": {
            "missing_values_X_train": int(np.isnan(X_train).sum()),
            "missing_values_X_test": int(np.isnan(X_test).sum()),
        },
        "errors": errors,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=4), encoding="utf-8")

    if errors:
        raise ValueError(
            "Preprocessing output validation failed: " + "; ".join(errors)
        )

    print("[SUCCESS] Output validation passed.")
    print(f"[INFO] Summary report saved to {REPORT_PATH}")


if __name__ == "__main__":
    validate_preprocessing_outputs()
