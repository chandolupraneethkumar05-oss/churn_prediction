import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "artifacts" / "production_model_report.json"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.pkl"
MLFLOW_DATABASE = PROJECT_ROOT / "mlflow.db"


def generate_registry_report():
    print("[INFO] Generating Model Registry Report...")

    client = MlflowClient(tracking_uri=f"sqlite:///{MLFLOW_DATABASE.as_posix()}")
    model_name = "Telco_Churn_Production_Model"

    try:
        # Search for the model specifically in Production
        production_models = client.search_model_versions(f"name='{model_name}'")
        champion = next((mv for mv in production_models if mv.current_stage == "Production"), None)

        if not champion:
            raise RuntimeError("No model found in Production stage!")

        # Get the run details to extract metrics and parameters
        run = client.get_run(champion.run_id)

        # Build the deployment-ready artifact dictionary
        report = {
            "registry_status": "READY_FOR_DEPLOYMENT",
            "model_lineage": {
                "registered_name": model_name,
                "version": int(champion.version),
                "current_stage": champion.current_stage,
                "run_id": champion.run_id,
                "artifact_uri": champion.source
            },
            "performance_metrics": {
                "recall": run.data.metrics.get("recall"),
                "f1_score": run.data.metrics.get("f1_score"),
                "roc_auc": run.data.metrics.get("roc_auc"),
                "accuracy": run.data.metrics.get("accuracy"),
                "precision": run.data.metrics.get("precision")
            },
            "hyperparameters": run.data.params,
            "preprocessing_dependency": str(PREPROCESSOR_PATH.relative_to(PROJECT_ROOT))
        }

        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, indent=4), encoding="utf-8")

        print(f"[SUCCESS] Report successfully generated for Version {champion.version}")
        print(f"[INFO] Saved to: {REPORT_PATH}")

    except Exception as e:
        raise RuntimeError(f"Failed to generate report: {e}") from e

if __name__ == "__main__":
    generate_registry_report()
