"""Train and evaluate a regression model for transaction quantity.

Run ``python data_prep.py`` first, then ``python ml_model.py``.
"""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data_prep import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, ID_COLUMN, NUMERIC_COLUMNS, TARGET

BASE_DIR = Path(__file__).resolve().parent


def build_pipeline() -> Pipeline:
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_COLUMNS),
        ("cat", categorical_pipe, CATEGORICAL_COLUMNS),
    ])
    model = RandomForestRegressor(
        n_estimators=250,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def main() -> None:
    train_path, test_path = BASE_DIR / "sales_train.csv", BASE_DIR / "sales_test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Jalankan `python data_prep.py` terlebih dahulu.")
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    pipeline = build_pipeline()
    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET]
    X_test, y_test = test[FEATURE_COLUMNS], test[TARGET]

    print("Training Random Forest untuk memprediksi jumlah produk per transaksi...")
    pipeline.fit(X_train, y_train)
    prediction = np.maximum(pipeline.predict(X_test), 0)
    metrics = {
        "mae": float(mean_absolute_error(y_test, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, prediction))),
        "r2": float(r2_score(y_test, prediction)),
        "test_rows": int(len(test)),
        "target_mean": float(y_test.mean()),
        "target_median": float(y_test.median()),
        "test_start_date": str(pd.to_datetime(test["transaction_date"]).min().date()),
    }
    print(f"MAE : {metrics['mae']:.3f} unit")
    print(f"RMSE: {metrics['rmse']:.3f} unit")
    print(f"R²  : {metrics['r2']:.3f}")

    joblib.dump(pipeline, BASE_DIR / "coffee_sales_pipeline.joblib")
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    importance = pipeline.named_steps["model"].feature_importances_
    pd.DataFrame({"feature": names, "importance": importance}).sort_values(
        "importance", ascending=False
    ).to_csv(BASE_DIR / "feature_importance.csv", index=False)
    output = test[[ID_COLUMN, "transaction_date", TARGET]].copy()
    output["predicted_qty"] = prediction
    output.to_csv(BASE_DIR / "sales_predictions.csv", index=False)
    (BASE_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Tersimpan: coffee_sales_pipeline.joblib, feature_importance.csv, sales_predictions.csv, model_metrics.json")


if __name__ == "__main__":
    main()
