"""Streamlit dashboard for the Student Dropout and Academic Success dataset.

Run ``python data_prep.py``, ``python ml_model.py``, then
``streamlit run app.py`` from this directory.
"""
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

from data_prep import ID_COLUMN, NUMERIC_COLUMNS, TARGET

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "studentdoa.csv"
MODEL_FILE = BASE_DIR / "student_outcome_pipeline.joblib"

st.set_page_config(page_title="Student Outcome Explorer", layout="wide")
st.title("Student Outcome Explorer")
st.caption("Eksplorasi data dan klasifikasi hasil studi mahasiswa: Dropout, Enrolled, atau Graduate.")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE) if MODEL_FILE.exists() else None


@st.cache_data
def load_optional_outputs():
    fi_path, metrics_path = BASE_DIR / "feature_importance.csv", BASE_DIR / "model_metrics.json"
    fi = pd.read_csv(fi_path) if fi_path.exists() else pd.DataFrame()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    return fi, metrics


df = load_data()
model = load_model()
feature_importance, metrics = load_optional_outputs()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Jumlah mahasiswa", f"{len(df):,}")
col2.metric("Jumlah fitur", f"{len(df.columns) - 1}")
col3.metric("Jumlah kelas", str(df[TARGET].nunique()))
col4.metric("Nilai kosong", f"{int(df.isna().sum().sum()):,}")

tab_overview, tab_explore, tab_model = st.tabs(["Ringkasan data", "Eksplorasi mahasiswa", "Machine learning"])

with tab_overview:
    left, right = st.columns(2)
    with left:
        st.subheader("Distribusi target")
        counts = df[TARGET].value_counts().rename_axis("Hasil studi").to_frame("Jumlah")
        st.bar_chart(counts)
        st.dataframe(counts, use_container_width=True)
    with right:
        st.subheader("Cuplikan dataset")
        st.dataframe(df.head(30), use_container_width=True)
        st.caption(f"Dataset berisi {df.shape[0]:,} baris dan {df.shape[1]} kolom.")

    st.info(
        "Catatan interpretasi: fitur semester 2 termasuk dalam model. Karena itu, model ini "
        "mengklasifikasikan hasil dengan informasi akademik semester 2 yang sudah tersedia; "
        "hasilnya tidak boleh dibaca sebagai prediksi saat mahasiswa baru mendaftar."
    )

with tab_explore:
    st.subheader("Lihat profil baris data")
    row_index = st.selectbox("Pilih mahasiswa (nomor baris dataset)", range(len(df)), format_func=lambda i: f"Mahasiswa {i + 1}")
    row = df.iloc[row_index]
    st.metric("Target aktual", str(row[TARGET]))
    profile = row.drop(labels=[TARGET]).rename_axis("Fitur").reset_index(name="Nilai")
    st.dataframe(profile, use_container_width=True, hide_index=True)

    st.subheader("Perbandingan fitur numerik")
    numeric_feature = st.selectbox("Pilih fitur", NUMERIC_COLUMNS, index=NUMERIC_COLUMNS.index("Age at enrollment"))
    st.bar_chart(df.groupby(TARGET)[numeric_feature].mean().sort_values(ascending=False))

with tab_model:
    st.subheader("Klasifikasi hasil studi")
    if model is None:
        st.warning("Model belum tersedia. Jalankan `python data_prep.py`, lalu `python ml_model.py`.")
    else:
        feature_columns = [column for column in df.columns if column != TARGET]
        X = df.loc[[row_index], feature_columns]
        prediction = model.predict(X)[0]
        st.metric("Prediksi model", str(prediction))
        st.caption(f"Target aktual: {row[TARGET]}")
        if hasattr(model, "predict_proba"):
            probabilities = pd.Series(model.predict_proba(X)[0], index=model.classes_, name="Probabilitas")
            st.markdown("**Probabilitas per kelas**")
            st.bar_chart(probabilities)

        if metrics:
            a, b, c = st.columns(3)
            a.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            b.metric("Macro F1", f"{metrics['macro_f1']:.3f}")
            c.metric("Weighted F1", f"{metrics['weighted_f1']:.3f}")
            st.markdown("**Confusion matrix** (baris = aktual, kolom = prediksi)")
            matrix = pd.DataFrame(metrics["confusion_matrix"], index=metrics["labels"], columns=metrics["labels"])
            st.dataframe(matrix, use_container_width=True)

        if not feature_importance.empty:
            st.markdown("**15 fitur dengan importance tertinggi**")
            st.bar_chart(feature_importance.head(15).set_index("feature")["importance"])

st.caption("Nilai importance dan probabilitas membantu eksplorasi, tetapi bukan bukti sebab-akibat atau kepastian hasil individu.")
