"""Prepare the Student Dropout and Academic Success dataset.

Run from this directory with ``python data_prep.py``. Creates stratified
train/test CSVs and a small data quality report beside the source CSV.
"""
from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
SOURCE_FILE = BASE_DIR / "studentdoa.csv"
TARGET = "Target"
ID_COLUMN = "student_id"

# These fields are numeric measurements/counts. The remaining predictors are
# integer-coded categories and should not be treated as ordered quantities.
NUMERIC_COLUMNS = [
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]


def load_data(path: Path = SOURCE_FILE) -> pd.DataFrame:
    """Load and validate the expected dataset schema."""
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Kolom target {TARGET!r} tidak ditemukan.")
    missing = set(NUMERIC_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Kolom numerik yang diwajibkan tidak ditemukan: {sorted(missing)}")
    if df[TARGET].isna().any():
        raise ValueError("Target berisi nilai kosong; baris tersebut perlu ditinjau sebelum training.")
    if df[TARGET].nunique() < 2:
        raise ValueError("Target harus memiliki setidaknya dua kelas.")
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create stable row IDs and keep target labels as readable class names."""
    result = df.copy()
    result.insert(0, ID_COLUMN, [f"STU-{i:04d}" for i in range(1, len(result) + 1)])
    result[TARGET] = result[TARGET].astype(str).str.strip()
    return result


def main() -> None:
    raw = load_data()
    model_df = prepare_features(raw)
    train_df, test_df = train_test_split(
        model_df,
        test_size=0.20,
        random_state=42,
        stratify=model_df[TARGET],
    )
    train_df.to_csv(BASE_DIR / "student_train.csv", index=False)
    test_df.to_csv(BASE_DIR / "student_test.csv", index=False)
    model_df.to_csv(BASE_DIR / "student_ml_dataset.csv", index=False)

    report = {
        "source_file": SOURCE_FILE.name,
        "rows": int(len(raw)),
        "columns": int(raw.shape[1]),
        "missing_cells": int(raw.isna().sum().sum()),
        "duplicate_rows": int(raw.duplicated().sum()),
        "class_counts": {str(k): int(v) for k, v in raw[TARGET].value_counts().items()},
        "class_percentages": {str(k): round(float(v * 100), 2) for k, v in raw[TARGET].value_counts(normalize=True).items()},
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_count": int(len(raw.columns) - 1),
        "numeric_feature_count": len(NUMERIC_COLUMNS),
        "note": "Kolom semester 2 dipakai sebagai fitur; prediksi ini menggunakan informasi setelah semester 2, bukan prediksi saat pendaftaran.",
    }
    (BASE_DIR / "data_quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Data preparation selesai.")
    print(f"Baris: {len(raw):,} | Train/test: {len(train_df):,}/{len(test_df):,}")
    print(f"Kelas target: {', '.join(raw[TARGET].value_counts().index)}")
    print("Dibuat: student_train.csv, student_test.csv, student_ml_dataset.csv, data_quality_report.json")


if __name__ == "__main__":
    main()
