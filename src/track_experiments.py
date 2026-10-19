
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, f1_score, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "heart_disease_clean.csv"
ARTIFACTS = ROOT / "reports" / "mlflow_artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

DB_PATH = ROOT / "mlflow.db"
mlflow.set_tracking_uri(
    f"sqlite:///{DB_PATH.as_posix()}"
)
mlflow.set_experiment("Heart-Disease-Classification")


def main():
    df = pd.read_csv(DATA)
    X = df.drop(columns=["target"])
    y = df["target"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=2000, random_state=42
            )),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=200, max_depth=5,
                random_state=42, class_weight="balanced"
            )),
        ]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc",
    }

    for name, pipeline in models.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model", name)
            mlflow.log_param("test_size", 0.20)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("cv_folds", 5)
            mlflow.log_param("preprocessing", "median imputation")
            mlflow.log_param("feature_count", X.shape[1])

            classifier = pipeline.named_steps["classifier"]
            if name == "Logistic Regression":
                mlflow.log_param("max_iter", classifier.max_iter)
                mlflow.log_param("scaling", "StandardScaler")
            else:
                mlflow.log_param("n_estimators", classifier.n_estimators)
                mlflow.log_param("max_depth", classifier.max_depth)
                mlflow.log_param("class_weight", "balanced")

            cv_results = cross_validate(
                pipeline, X_train, y_train,
                cv=cv, scoring=scoring, n_jobs=-1
            )

            for metric in scoring:
                mlflow.log_metric(
                    f"cv_{metric}_mean",
                    float(cv_results[f"test_{metric}"].mean())
                )

            model = clone(pipeline).fit(X_train, y_train)
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)[:, 1]

            metrics = {
                "test_accuracy": accuracy_score(y_test, predictions),
                "test_precision": precision_score(
                    y_test, predictions, zero_division=0
                ),
                "test_recall": recall_score(
                    y_test, predictions, zero_division=0
                ),
                "test_f1": f1_score(
                    y_test, predictions, zero_division=0
                ),
                "test_roc_auc": roc_auc_score(y_test, probabilities),
            }

            mlflow.log_metrics({
                k: float(v) for k, v in metrics.items()
            })

            report = classification_report(
                y_test, predictions, zero_division=0
            )
            report_path = ARTIFACTS / (
                name.lower().replace(" ", "_") + "_report.txt"
            )
            report_path.write_text(report, encoding="utf-8")
            mlflow.log_artifact(str(report_path))

            ConfusionMatrixDisplay.from_predictions(
                y_test, predictions,
                display_labels=["No disease", "Disease"]
            )
            plt.title(f"{name} - Confusion Matrix")
            plt.tight_layout()
            cm_path = ARTIFACTS / (
                name.lower().replace(" ", "_") + "_confusion.png"
            )
            plt.savefig(cm_path, dpi=150)
            plt.close()
            mlflow.log_artifact(str(cm_path))

            RocCurveDisplay.from_predictions(y_test, probabilities)
            plt.title(f"{name} - ROC Curve")
            plt.tight_layout()
            roc_path = ARTIFACTS / (
                name.lower().replace(" ", "_") + "_roc.png"
            )
            plt.savefig(roc_path, dpi=150)
            plt.close()
            mlflow.log_artifact(str(roc_path))

            mlflow.sklearn.log_model(
    model,
    name="model",
    serialization_format="cloudpickle",
)

            print(f"\nLogged run: {name}")
            print(metrics)

    print("\nMLflow experiment tracking completed.")
    print("Run: python -m mlflow ui --backend-store-uri ./mlruns")


if __name__ == "__main__":
    main()
