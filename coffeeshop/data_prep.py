"""Prepare Coffee Shop Sales transactions for quantity prediction.

Run ``python data_prep.py`` from this folder. The source is the supplied
``Coffee Shop Sales.xlsx`` workbook; generated CSVs are written beside it.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
SOURCE_FILE = BASE_DIR / "Coffee Shop Sales.xlsx"
TARGET = "transaction_qty"
ID_COLUMN = "transaction_id"

RAW_COLUMNS = [
    "transaction_id", "transaction_date", "transaction_time", "transaction_qty",
    "store_id", "store_location", "product_id", "unit_price",
    "product_category", "product_type", "product_detail",
]
CATEGORICAL_COLUMNS = [
    "store_id", "store_location", "product_id", "product_category",
    "product_type", "product_detail",
]
NUMERIC_COLUMNS = ["unit_price", "year", "month", "day", "day_of_week", "hour"]
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


def load_source_data(path: Path = SOURCE_FILE) -> pd.DataFrame:
    """Read the workbook and fail early if its expected schema changed."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")
    df = pd.read_excel(path, engine="openpyxl")
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Kolom dataset tidak ditemukan: {sorted(missing)}")
    return df[RAW_COLUMNS].copy()


def prepare_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Clean source fields and derive calendar features from date and time."""
    result = df.copy()
    # Excel stores dates and times as day counts/fractions; pandas usually
    # converts the date cell automatically but the time may remain numeric.
    date_values = result["transaction_date"]
    if pd.api.types.is_numeric_dtype(date_values):
        result["transaction_date"] = pd.to_datetime(
            date_values, unit="D", origin="1899-12-30", errors="coerce"
        )
    else:
        result["transaction_date"] = pd.to_datetime(date_values, errors="coerce")
    time_values = result["transaction_time"]
    if pd.api.types.is_numeric_dtype(time_values):
        result["transaction_time"] = pd.to_timedelta(time_values, unit="D", errors="coerce")
        result["hour"] = (result["transaction_time"].dt.total_seconds() // 3600).astype("Int64")
    else:
        parsed_time = pd.to_datetime(time_values.astype(str), errors="coerce")
        result["hour"] = parsed_time.dt.hour.astype("Int64")

    result["transaction_qty"] = pd.to_numeric(result[TARGET], errors="coerce")
    result["unit_price"] = pd.to_numeric(result["unit_price"], errors="coerce")
    result["transaction_date"] = result["transaction_date"].dt.normalize()
    result["year"] = result["transaction_date"].dt.year.astype("Int64")
    result["month"] = result["transaction_date"].dt.month.astype("Int64")
    result["day"] = result["transaction_date"].dt.day.astype("Int64")
    result["day_of_week"] = result["transaction_date"].dt.dayofweek.astype("Int64")

    for column in CATEGORICAL_COLUMNS:
        result[column] = result[column].astype("string").str.strip()
    result = result.dropna(subset=[TARGET, "transaction_date", "unit_price", "hour"])
    result = result[result[TARGET] > 0].copy()
    result[ID_COLUMN] = result[ID_COLUMN].astype(str)
    return result.sort_values(["transaction_date", ID_COLUMN]).reset_index(drop=True)


def main() -> None:
    raw = load_source_data()
    data = prepare_transactions(raw)
    if len(data) < 10:
        raise ValueError("Jumlah baris valid terlalu sedikit untuk train/test split.")

    # Hold out the latest 20% of transactions to respect the time direction.
    split_at = int(len(data) * 0.80)
    train_df, test_df = data.iloc[:split_at].copy(), data.iloc[split_at:].copy()
    if train_df[TARGET].nunique() < 1 or test_df.empty:
        raise ValueError("Split kronologis gagal; periksa rentang tanggal dataset.")

    output_columns = [ID_COLUMN, "transaction_date", *FEATURE_COLUMNS, TARGET]
    train_df[output_columns].to_csv(BASE_DIR / "sales_train.csv", index=False)
    test_df[output_columns].to_csv(BASE_DIR / "sales_test.csv", index=False)
    data[output_columns].to_csv(BASE_DIR / "sales_ml_dataset.csv", index=False)

    report = {
        "source_file": SOURCE_FILE.name,
        "source_rows": int(len(raw)),
        "prepared_rows": int(len(data)),
        "removed_rows": int(len(raw) - len(data)),
        "duplicate_transaction_ids": int(data[ID_COLUMN].duplicated().sum()),
        "date_min": str(data["transaction_date"].min().date()),
        "date_max": str(data["transaction_date"].max().date()),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_end_date": str(train_df["transaction_date"].max().date()),
        "test_start_date": str(test_df["transaction_date"].min().date()),
        "target": TARGET,
        "target_mean": round(float(data[TARGET].mean()), 3),
        "target_median": float(data[TARGET].median()),
        "target_max": int(data[TARGET].max()),
        "missing_values_after_prep": int(data[output_columns].isna().sum().sum()),
        "note": "Model memprediksi jumlah unit per transaksi; pengujian memakai transaksi dengan tanggal paling akhir.",
    }
    (BASE_DIR / "data_quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Data preparation selesai.")
    print(f"Baris valid: {len(data):,} | Train/test: {len(train_df):,}/{len(test_df):,}")
    print(f"Periode train/test: sampai {report['train_end_date']} / mulai {report['test_start_date']}")
    print("Dibuat: sales_ml_dataset.csv, sales_train.csv, sales_test.csv, data_quality_report.json")


if __name__ == "__main__":
    main()
