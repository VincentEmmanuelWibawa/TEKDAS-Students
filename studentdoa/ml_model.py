"""Train and evaluate a multiclass student outcome model.

Run ``python data_prep.py`` first, then ``python ml_model.py``.
"""
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data_prep import ID_COLUMN, NUMERIC_COLUMNS, TARGET

BASE_DIR = Path(__file__).resolve().parent


def build_pipeline(categorical_columns):
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_COLUMNS),
        ("cat", categorical_pipe, categorical_columns),
    ])
    model = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def main() -> None:
    train_path, test_path = BASE_DIR / "student_train.csv", BASE_DIR / "student_test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Jalankan `python data_prep.py` terlebih dahulu.")
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    feature_columns = [c for c in train.columns if c not in (ID_COLUMN, TARGET)]
    categorical_columns = [c for c in feature_columns if c not in NUMERIC_COLUMNS]
    pipeline = build_pipeline(categorical_columns)
    X_train, y_train = train[feature_columns], train[TARGET].astype(str)
    X_test, y_test = test[feature_columns], test[TARGET].astype(str)

    print("Training Random Forest untuk klasifikasi hasil studi...")
    pipeline.fit(X_train, y_train)
    prediction = pipeline.predict(X_test)
    labels = sorted(set(y_test) | set(prediction))
    metrics = {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "macro_f1": float(f1_score(y_test, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, prediction, average="weighted", zero_division=0)),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_test, prediction, labels=labels).tolist(),
        "test_rows": int(len(test)),
    }
    print("\nClassification report:\n")
    print(classification_report(y_test, prediction, zero_division=0))

    joblib.dump(pipeline, BASE_DIR / "student_outcome_pipeline.joblib")
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    importance = pipeline.named_steps["model"].feature_importances_
    pd.DataFrame({"feature": names, "importance": importance}).sort_values(
        "importance", ascending=False
    ).to_csv(BASE_DIR / "feature_importance.csv", index=False)
    output = test[[ID_COLUMN, TARGET]].copy()
    output["predicted_target"] = prediction
    output.to_csv(BASE_DIR / "student_predictions.csv", index=False)
    (BASE_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Tersimpan: student_outcome_pipeline.joblib, feature_importance.csv, student_predictions.csv, model_metrics.json")


if __name__ == "__main__":
    main()
