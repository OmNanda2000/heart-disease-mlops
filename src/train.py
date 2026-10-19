from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import pandas as pd

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "heart_disease_clean.csv"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "reports"
PLOT_DIR = ROOT / "screenshots" / "models"
MLFLOW_DIR = ROOT / "mlruns"
MLFLOW_DB_PATH = ROOT / "mlflow.db"
EXPERIMENT_NAME = "Heart Disease Prediction"

for folder in [MODEL_DIR, RESULT_DIR, PLOT_DIR, MLFLOW_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


def main():
    # Use SQLite for MLflow tracking metadata instead of the deprecated file-store backend.
    tracking_uri = f"sqlite:///{MLFLOW_DB_PATH.resolve().as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("Loading cleaned dataset...")
    df = pd.read_csv(DATA_PATH)

    if "target" not in df.columns:
        raise ValueError("The dataset must contain a 'target' column.")

    X = df.drop(columns=["target"])
    y = df["target"].astype(int)

    # Keep the test set separate from model selection.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    # Imputation is fitted inside each training fold.
    # Logistic Regression also needs feature scaling.
    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=2000,
                random_state=42,
            )),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                max_depth=5,
                random_state=42,
                class_weight="balanced",
            )),
        ]),
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc",
    }

    results = []
    fitted_models = {}
    cv_means = {}
    run_ids = {}

    for name, pipeline in models.items():
        print(f"\nEvaluating {name}...")

        with mlflow.start_run(run_name=name) as active_run:
            run_ids[name] = active_run.info.run_id

            classifier = pipeline.named_steps["classifier"]
            mlflow.log_params({
                "model_name": name,
                "cv_folds": 5,
                "test_size": 0.20,
                "random_state": 42,
                **{
                    f"classifier_{key}": value
                    for key, value in classifier.get_params(deep=False).items()
                    if isinstance(value, (str, int, float, bool))
                    or value is None
                },
            })

            cv_results = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
            )

            cv_means[name] = float(cv_results["test_roc_auc"].mean())

            print("Cross-validation ROC-AUC:",
                  round(cv_means[name], 4))

            fitted = clone(pipeline)
            fitted.fit(X_train, y_train)
            fitted_models[name] = fitted

            y_pred = fitted.predict(X_test)
            y_prob = fitted.predict_proba(X_test)[:, 1]

            metrics = {
                "model": name,
                "cv_accuracy_mean":
                    float(cv_results["test_accuracy"].mean()),
                "cv_precision_mean":
                    float(cv_results["test_precision"].mean()),
                "cv_recall_mean":
                    float(cv_results["test_recall"].mean()),
                "cv_roc_auc_mean": cv_means[name],
                "test_accuracy": float(accuracy_score(y_test, y_pred)),
                "test_precision": float(
                    precision_score(y_test, y_pred, zero_division=0)
                ),
                "test_recall": float(
                    recall_score(y_test, y_pred, zero_division=0)
                ),
                "test_f1": float(
                    f1_score(y_test, y_pred, zero_division=0)
                ),
                "test_roc_auc": float(roc_auc_score(y_test, y_prob)),
            }
            results.append(metrics)

            mlflow.log_metrics({
                key: value
                for key, value in metrics.items()
                if key != "model"
            })

            print("\nTest metrics:")
            for metric, value in metrics.items():
                if metric != "model":
                    print(f"{metric}: {value:.4f}")

            report = classification_report(
                y_test,
                y_pred,
                zero_division=0,
            )
            print("\nClassification report:")
            print(report)

            report_path = RESULT_DIR / (
                name.lower().replace(" ", "_") + "_classification_report.txt"
            )
            report_path.write_text(report, encoding="utf-8")
            mlflow.log_artifact(str(report_path), artifact_path="reports")

            # Confusion matrix
            confusion_path = PLOT_DIR / (
                name.lower().replace(" ", "_") + "_confusion_matrix.png"
            )
            ConfusionMatrixDisplay.from_predictions(
                y_test,
                y_pred,
                display_labels=["No disease", "Disease"],
            )
            plt.title(f"{name} - Confusion Matrix")
            plt.tight_layout()
            plt.savefig(confusion_path, dpi=200)
            plt.close()
            mlflow.log_artifact(str(confusion_path), artifact_path="plots")

            # ROC curve
            roc_path = PLOT_DIR / (
                name.lower().replace(" ", "_") + "_roc_curve.png"
            )
            RocCurveDisplay.from_predictions(y_test, y_prob)
            plt.title(f"{name} - ROC Curve")
            plt.tight_layout()
            plt.savefig(roc_path, dpi=200)
            plt.close()
            mlflow.log_artifact(str(roc_path), artifact_path="plots")

            # Log the fitted sklearn pipeline with a model signature.
         # Log the fitted sklearn pipeline with a model signature.
        signature = infer_signature(X_train, fitted.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=fitted,
            name="trained_model",
            signature=signature,
            serialization_format="pickle",
        )

    results_df = pd.DataFrame(results)
    comparison_path = RESULT_DIR / "model_comparison.csv"
    results_df.to_csv(comparison_path, index=False)

    # Choose using training cross-validation ROC-AUC only.
    # The test set is not used to select the winning model.
    best_name = max(cv_means, key=cv_means.get)
    best_model = fitted_models[best_name]

    model_path = MODEL_DIR / "heart_disease_model.joblib"
    joblib.dump(best_model, model_path)

    model_info_path = MODEL_DIR / "model_info.json"
    with open(model_info_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": best_name,
                "features": list(X.columns),
                "target_mapping": {
                    "0": "No heart disease",
                    "1": "Heart disease",
                },
            },
            f,
            indent=2,
        )

    # Attach shared comparison/selection artifacts to both model runs.
    for name in models:
        mlflow.log_artifact(str(comparison_path), run_id=run_ids[name])
        if name == best_name:
            mlflow.log_artifact(str(model_path), run_id=run_ids[name],
                                artifact_path="selected_model_files")
            mlflow.log_artifact(str(model_info_path), run_id=run_ids[name],
                                artifact_path="selected_model_files")
            client.set_tag(run_ids[name], "selected_model", "true")
        else:
            client.set_tag(run_ids[name], "selected_model", "false")

    print("\nModel comparison:")
    print(results_df.round(4).to_string(index=False))
    print("\nSelected using training CV ROC-AUC:", best_name)
    print("Saved model:", model_path)
    print("Saved model info:", model_info_path)
    print("Saved metrics:", comparison_path)
    print("Saved plots:", PLOT_DIR)
    print("MLflow database:", MLFLOW_DB_PATH)
    print("\nTASK 2 COMPLETED WITH MLFLOW!")


if __name__ == "__main__":
    main()
