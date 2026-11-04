from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Smart Hydroponic Forecasting Dashboard",
    page_icon="🌿",
    layout="wide",
)

# --- 2. AUTO-REFRESH OTOMATIS (TIAP 5 DETIK) ---
# Halaman akan memuat ulang data & prediksi secara otomatis tanpa F5
count = st_autorefresh(interval=5000, limit=None, key="realtime_hydro_refresh")


# --- 3. LOAD MODEL & SCALER YANG SUDAH DILATIH ---
@st.cache_resource
def load_ml_assets():
  scaler = joblib.load("scaler.pkl")
  model_tds = joblib.load("model_tds.pkl")
  model_ph = joblib.load("model_ph.pkl")
  model_water = joblib.load("model_water.pkl")

  enc_tds = joblib.load("tds_encoder.pkl")
  enc_ph = joblib.load("ph_encoder.pkl")
  enc_water = joblib.load("water_encoder.pkl")

  return (
      scaler,
      model_tds,
      model_ph,
      model_water,
      enc_tds,
      enc_ph,
      enc_water,
  )


scaler, model_tds, model_ph, model_water, enc_tds, enc_ph, enc_water = (
    load_ml_assets()
)


# --- 4. FUNGSI AMBIL DATA LIVE DARI URL CSV ---
@st.cache_data(ttl=3)  # Cache reset tiap 3 detik
def fetch_live_csv_data():
  try:
    # ⚠️ GANTI LINK DI BAWAH INI DENGAN RAW URL CSV GITHUB KAMU ⚠️
    csv_url = "https://raw.githubusercontent.com/winnycintania2303226-art/wins/refs/heads/main/Gabungan%2012%20file%20100775%20data.csv"
    df = pd.read_csv(csv_url)
    return df
  except Exception as e:
    # Data fallback (jika URL belum diganti atau gagal diakses)
    return pd.DataFrame({
        "temperature": [28.90],
        "humidity": [77.0],
        "distance": [32.5],
        "tds": [1351.2],
        "ph": [10.69],
    })


# Ambil dataframe terbaru & baris paling akhir (data sensor detik ini)
df_live = fetch_live_csv_data()
latest = df_live.iloc[-1]

# --- 5. PROSES PREDIKSI MACHINE LEARNING OTOMATIS ---
# Susun fitur sesuai urutan saat training: ["temperature", "humidity", "distance", "tds", "ph"]
X_new = pd.DataFrame([{
    "temperature": latest["temperature"],
    "humidity": latest["humidity"],
    "distance": latest["distance"],
    "tds": latest["tds"],
    "ph": latest["ph"],
}])

# Scaling data menggunakan scaler yang sama saat training
X_new_scaled = scaler.transform(X_new)

# Prediksi model
pred_tds_idx = model_tds.predict(X_new_scaled)[0]
pred_ph_idx = model_ph.predict(X_new_scaled)[0]
pred_water_idx = model_water.predict(X_new_scaled)[0]

# Ubah angka hasil prediksi kembali ke label teks ("Normal", "Tinggi", "Rendah")
status_tds = enc_tds.inverse_transform([pred_tds_idx])[0]
status_ph = enc_ph.inverse_transform([pred_ph_idx])[0]
status_water = enc_water.inverse_transform([pred_water_idx])[0]


# --- 6. SIDEBAR (STATUS SENSOR) ---
st.sidebar.markdown("## 🌿 Smart Hydroponic")
st.sidebar.success("🟢 System Online & Live")
st.sidebar.markdown("---")
st.sidebar.markdown("### Status Prediksi ML")
st.sidebar.markdown(
    f"🧪 **TDS Status:** {'🔴' if status_tds != 'Normal' else '🟢'}"
    f" {status_tds}"
)
st.sidebar.markdown(
    f"⚗️ **pH Status:** {'🔴' if status_ph != 'Normal' else '🟢'} {status_ph}"
)
st.sidebar.markdown(
    f"🌊 **Water Level:** {'🟡' if status_water != 'Normal' else '🟢'}"
    f" {status_water}"
)


# --- 7. MAIN PAGE HEADER ---
st.title("🌿 Smart Hydroponic Forecasting Dashboard")
st.markdown("**AI-Based Real-Time Monitoring & Classification System**")
st.info(
    f"🟢 Live Connected | Auto-refresh count: {count} | Last Update:"
    f" {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"
)

st.markdown("---")

# --- 8. CURRENT SENSOR VS PREDIKSI ML ---
col1, col2 = st.columns(2)

with col1:
  st.subheader("📡 Current Sensor (Live)")
  c1, c2 = st.columns(2)
  c1.metric("🌡️ Temperature", f"{latest['temperature']:.2f} °C")
  c2.metric("💧 Humidity", f"{latest['humidity']:.1f} %")
  c1.metric("🧪 TDS", f"{latest['tds']:.1f} ppm")
  c2.metric("⚗️ pH", f"{latest['ph']:.2f}")
  st.metric("🌊 Distance / Water Level", f"{latest['distance']} cm")

with col2:
  st.subheader("🔮 AI / ML Classification Result")
  f1, f2 = st.columns(2)
  f1.metric("🧪 TDS Prediction", status_tds)
  f2.metric("⚗️ pH Prediction", status_ph)
  st.metric("🌊 Water Level Prediction", status_water)

st.markdown("---")

# --- 9. GRAFIK TREN DATA LIVE ---
st.subheader("📈 Sensor Trend (Real-Time)")
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Temperature", "Humidity", "TDS", "pH", "Distance"]
)

with tab1:
  st.line_chart(df_live["temperature"])
with tab2:
  st.line_chart(df_live["humidity"])
with tab3:
  st.line_chart(df_live["tds"])
with tab4:
  st.line_chart(df_live["ph"])
with tab5:
  st.line_chart(df_live["distance"])