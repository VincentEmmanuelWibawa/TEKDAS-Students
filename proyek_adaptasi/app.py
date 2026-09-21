
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

from config import DATA_FILENAME, FEATURE_COLUMNS, ID_COLUMN, TARGET
from llm import build_prompt, generate

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw" / DATA_FILENAME
MODEL_PATH = BASE_DIR / "artifacts" / "model.joblib"
METRICS_PATH = BASE_DIR / "artifacts" / "metrics.json"

st.set_page_config(page_title="Proyek Adaptasi", layout="wide")
st.title("TODO — Judul Aplikasi")
st.caption("TODO — siapa pengguna aplikasi dan keputusan apa yang dibantu")


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not DATA_PATH.exists():
    st.error("Dataset belum tersedia. Lengkapi config.py dan data/raw/.")
    st.stop()
if not MODEL_PATH.exists() or not METRICS_PATH.exists():
    st.error("Model belum tersedia. Jalankan data_prep.py lalu train.py.")
    st.stop()

data = load_data()
model = load_model()
metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

tab_data, tab_bi, tab_ml, tab_llm = st.tabs(["Data", "BI", "ML", "LLM"])

with tab_data:
    st.write(f"{len(data):,} baris × {len(data.columns)} kolom")
    st.dataframe(data.head(100), use_container_width=True)
    st.markdown("**Missing value per kolom**")
    st.dataframe(data.isna().sum().rename("missing"), use_container_width=True)

with tab_bi:
    st.info("TODO: ganti bagian ini dengan KPI/grafik yang menjawab pertanyaan stakeholder.")
    st.bar_chart(data[TARGET].value_counts())

with tab_ml:
    st.json(metrics)
    row_number = st.number_input("Nomor baris untuk diperiksa", 0, len(data) - 1, 0)
    selected = data.iloc[[row_number]]
    prediction = model.predict(selected[FEATURE_COLUMNS])[0]
    st.metric("Prediksi", str(prediction))
    if TARGET in selected:
        st.metric("Label aktual", str(selected.iloc[0][TARGET]))
    if ID_COLUMN and ID_COLUMN in selected:
        st.caption(f"{ID_COLUMN}: {selected.iloc[0][ID_COLUMN]}")
    st.dataframe(selected, use_container_width=True)

with tab_llm:
    row_number_llm = st.number_input("Nomor baris sebagai evidence", 0, len(data) - 1, 0, key="llm_row")
    record = data.iloc[row_number_llm][FEATURE_COLUMNS].to_dict()
    prediction = model.predict(data.iloc[[row_number_llm]][FEATURE_COLUMNS])[0]
    prompt = build_prompt(record, prediction, metrics["model"])
    with st.expander("Periksa prompt"):
        st.code(prompt, language="text")
    if st.button("Buat analisis grounded"):
        with st.spinner("Memanggil Ollama..."):
            st.write(generate(prompt))
