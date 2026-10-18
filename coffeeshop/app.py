"""Streamlit dashboard for Coffee Shop Sales.

Run ``python data_prep.py``, ``python ml_model.py``, then
``streamlit run app.py`` from this folder.
"""
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

from data_prep import FEATURE_COLUMNS, TARGET

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Coffee Shop Sales.xlsx"
MODEL_FILE = BASE_DIR / "coffee_sales_pipeline.joblib"

st.set_page_config(page_title="Coffee Shop Sales", layout="wide")
st.title("Coffee Shop Sales: Data, BI, dan Machine Learning")
st.caption("Eksplorasi transaksi dan prediksi jumlah unit produk yang terjual per transaksi.")


@st.cache_data
def load_data():
    df = pd.read_excel(DATA_FILE, engine="openpyxl")
    dates = df["transaction_date"]
    if pd.api.types.is_numeric_dtype(dates):
        df["transaction_date"] = pd.to_datetime(dates, unit="D", origin="1899-12-30", errors="coerce")
    else:
        df["transaction_date"] = pd.to_datetime(dates, errors="coerce")
    df["revenue"] = pd.to_numeric(df["transaction_qty"], errors="coerce") * pd.to_numeric(df["unit_price"], errors="coerce")
    return df


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

total_revenue = df["revenue"].sum()
total_units = df[TARGET].sum()
average_transaction = df["revenue"].mean()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Transaksi", f"{len(df):,}")
col2.metric("Total unit terjual", f"{total_units:,.0f}")
col3.metric("Estimasi revenue", f"${total_revenue:,.2f}")
col4.metric("Rata-rata transaksi", f"${average_transaction:,.2f}")

tab_bi, tab_explore, tab_ml = st.tabs(["Ringkasan penjualan", "Eksplorasi transaksi", "Machine learning"])

with tab_bi:
    left, right = st.columns(2)
    with left:
        st.subheader("Revenue per bulan")
        monthly = df.dropna(subset=["transaction_date"]).assign(
            month=lambda x: x["transaction_date"].dt.to_period("M").astype(str)
        ).groupby("month")["revenue"].sum()
        st.line_chart(monthly)
    with right:
        st.subheader("Revenue per kategori produk")
        category_sales = df.groupby("product_category")["revenue"].sum().sort_values(ascending=False)
        st.bar_chart(category_sales)

    st.subheader("Produk dengan revenue tertinggi")
    product_sales = (df.groupby(["product_category", "product_type"], as_index=False)
                     .agg(revenue=("revenue", "sum"), units=(TARGET, "sum"))
                     .sort_values("revenue", ascending=False).head(15))
    st.dataframe(product_sales, use_container_width=True, hide_index=True)

with tab_explore:
    st.subheader("Filter transaksi")
    stores = sorted(df["store_location"].dropna().unique().tolist())
    categories = sorted(df["product_category"].dropna().unique().tolist())
    f1, f2 = st.columns(2)
    selected_store = f1.selectbox("Lokasi toko", ["Semua"] + stores)
    selected_category = f2.selectbox("Kategori produk", ["Semua"] + categories)
    view = df.copy()
    if selected_store != "Semua":
        view = view[view["store_location"] == selected_store]
    if selected_category != "Semua":
        view = view[view["product_category"] == selected_category]
    st.write(f"Menampilkan {len(view):,} transaksi")
    st.dataframe(view.head(500), use_container_width=True, hide_index=True)

with tab_ml:
    st.subheader("Prediksi jumlah unit per transaksi")
    st.caption("Model menggunakan produk, toko, harga, tanggal, hari, dan jam. Evaluasi menggunakan 20% transaksi dengan tanggal paling akhir.")
    if model is None:
        st.warning("Model belum tersedia. Jalankan `python data_prep.py`, lalu `python ml_model.py`.")
    else:
        row_number = st.slider("Pilih baris transaksi untuk dicoba", 0, len(df) - 1, 0)
        raw_row = df.iloc[[row_number]]
        st.dataframe(raw_row, use_container_width=True, hide_index=True)

        # Match the calendar/time feature engineering used for training.
        sample = raw_row.copy()
        sample_date = sample["transaction_date"]
        if pd.api.types.is_numeric_dtype(sample_date):
            sample["transaction_date"] = pd.to_datetime(sample_date, unit="D", origin="1899-12-30", errors="coerce")
        else:
            sample["transaction_date"] = pd.to_datetime(sample_date, errors="coerce")
        sample["year"] = sample["transaction_date"].dt.year
        sample["month"] = sample["transaction_date"].dt.month
        sample["day"] = sample["transaction_date"].dt.day
        sample["day_of_week"] = sample["transaction_date"].dt.dayofweek
        if pd.api.types.is_numeric_dtype(sample["transaction_time"]):
            sample["hour"] = (pd.to_numeric(sample["transaction_time"], errors="coerce") * 24).astype(int)
        else:
            parsed_time = pd.to_datetime(sample["transaction_time"].astype(str), errors="coerce")
            sample["hour"] = parsed_time.dt.hour.fillna(0).astype(int)
        sample["unit_price"] = pd.to_numeric(sample["unit_price"], errors="coerce")
        for column in ["store_id", "store_location", "product_id", "product_category", "product_type", "product_detail"]:
            sample[column] = sample[column].astype("string").str.strip()
        X_sample = sample[FEATURE_COLUMNS]
        predicted_qty = float(model.predict(X_sample)[0])
        a, b = st.columns(2)
        a.metric("Prediksi jumlah unit", f"{predicted_qty:.2f}")
        b.metric("Jumlah aktual", f"{int(raw_row.iloc[0][TARGET])}")

        if metrics:
            m1, m2, m3 = st.columns(3)
            m1.metric("MAE", f"{metrics['mae']:.3f} unit")
            m2.metric("RMSE", f"{metrics['rmse']:.3f} unit")
            m3.metric("R²", f"{metrics['r2']:.3f}")
            st.caption(f"Data uji mulai {metrics['test_start_date']}; MAE mengukur rata-rata selisih prediksi dalam unit.")
        if not feature_importance.empty:
            st.markdown("**15 fitur dengan importance tertinggi**")
            st.bar_chart(feature_importance.head(15).set_index("feature")["importance"])

    st.info("Prediksi ini memperkirakan kuantitas transaksi berdasarkan data historis; ini bukan ramalan total penjualan harian.")
