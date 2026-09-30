import os
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def run_training():

    print("Starting MLflow Model Training...")

    # 1. Load processed training and test data
    X_train_final = np.load("data/processed/X_train_final.npy")
    y_train = np.load("data/processed/y_train.npy")

    X_test_final = np.load("data/processed/X_test_final.npy")
    y_test = np.load("data/processed/y_test.npy")

    # 2. Model configuration
    n_estimators = 100
    max_depth = 10
    random_state = 42
    class_weight = "balanced"

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight=class_weight
    )

    # 3. Start MLflow experiment
    mlflow.set_experiment("ChurnPrediction-Lab4")

    with mlflow.start_run():

        # Log parameters
        mlflow.log_param("model", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("class_weight", class_weight)

        # 4. Train
        model.fit(X_train_final, y_train)

        # 5. Predictions
        y_pred = model.predict(X_test_final)

        # 6. Evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # 7. Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # 8. Save model
        os.makedirs("models", exist_ok=True)

        model_path = "models/random_forest_mlflow.pkl"
        joblib.dump(model, model_path)

        # 9. Log model to MLflow
        try:
            mlflow.sklearn.log_model(
                model,
                name="random_forest_model",
                skops_trusted_types=["sklearn.tree._tree.Tree"]
            )
        except Exception:
            try:
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="random_forest_model",
                    skops_trusted_types=["sklearn.tree._tree.Tree"]
                )
            except Exception:
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="random_forest_model",
                    serialization_format="cloudpickle"
                )

        print("MLflow run completed successfully!")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    run_training()