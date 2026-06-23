"""
Lecture Note Praktikum: Pemodelan dan Simulasi
Minggu 14: Pembuatan Simulator Interaktif dan Analisis What-If
"""

# ============================================================
# Langkah 1: Persiapan Model dan Baseline
# ============================================================
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import scipy.stats as stats

# Menyiapkan data historis sederhana
# Fitur: [Iklan (Juta), Diskon (%)]
X_train = np.array([[5, 10], [10, 20], [15, 5], [20, 25], [25, 15]])
# Target: Keuntungan (Juta)
y_train = np.array([50, 80, 110, 90, 150])

# Melatih model (Mesin Replika)
model = LinearRegression().fit(X_train, y_train)

# ============================================================
# Langkah 1b: Statistik Deskriptif Data Historis
# ============================================================
df = pd.DataFrame(X_train, columns=['Iklan (Juta)', 'Diskon (%)'])
df['Keuntungan (Juta)'] = y_train

print("=" * 55)
print("STATISTIK DESKRIPTIF DATA HISTORIS")
print("=" * 55)
print(df.describe())
print()

# Matriks korelasi
print("-" * 55)
print("MATRIKS KORELASI PEARSON")
print("-" * 55)
corr_matrix = df.corr(method='pearson')
print(corr_matrix)
print()

# Uji signifikansi korelasi antara tiap fitur dan target
print("-" * 55)
print("UJI SIGNIFIKANSI KORELASI (H0: r = 0)")
print("-" * 55)
for col in ['Iklan (Juta)', 'Diskon (%)']:
    r, p_value = stats.pearsonr(df[col], df['Keuntungan (Juta)'])
    bintng = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    print(f"{col:20s}  r = {r:+.4f}  p = {p_value:.4f}  {bintng}")
print()

# ============================================================
# Langkah 1c: Evaluasi Model (Goodness-of-Fit)
# ============================================================
y_pred_train = model.predict(X_train)

r2 = r2_score(y_train, y_pred_train)
mae = mean_absolute_error(y_train, y_pred_train)
rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
mape = np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100

print("-" * 55)
print("METRIK EVALUASI MODEL REGRESI")
print("-" * 55)
print(f"R² (Koefisien Determinasi)  : {r2:.4f}")
print(f"MAE (Mean Absolute Error)   : {mae:.2f} Juta")
print(f"RMSE (Root MSE)             : {rmse:.2f} Juta")
print(f"MAPE (Mean Abs % Error)     : {mape:.2f} %")
print()

# Tampilkan koefisien model
print("-" * 55)
print("KOEFISIEN MODEL REGRESI")
print("-" * 55)
print(f"Intercept  : {model.intercept_:.4f}")
print(f"Koef Iklan : {model.coef_[0]:.4f}  (setiap +1 Juta iklan → +{model.coef_[0]:.2f} Juta)")
print(f"Koef Diskon: {model.coef_[1]:.4f}  (setiap +1% diskon → +{model.coef_[1]:.2f} Juta)")
print()

# ============================================================
# Langkah 2: Menetapkan Skenario Dasar (Baseline)
# ============================================================
# Kondisi saat ini: Iklan 10 Juta, Diskon 10%
baseline_input = np.array([[10, 10]])
baseline_pred = model.predict(baseline_input)[0]
print(f"Prediksi Keuntungan Baseline: Rp {baseline_pred:.2f} Juta")

# ============================================================
# Langkah 3: Membangun Logika Simulator (Analisis What-If)
# ============================================================
def run_simulation(new_iklan, new_diskon):
    # Input baru dari user (Intervensi)
    intervention_input = np.array([[new_iklan, new_diskon]])

    # Prediksi hasil intervensi
    prediction = model.predict(intervention_input)[0]

    # Menghitung Delta (Selisih)
    delta_y = prediction - baseline_pred

    return prediction, delta_y

# ============================================================
# Langkah 4: Implementasi UI Interaktif (Konsep Streamlit)
# ============================================================
# Catatan: Kode di bawah ini adalah struktur aplikasi Streamlit (app.py).
# Jalankan dengan: streamlit run nama_file_ini.py

import streamlit as st

st.title("Simulator Kebijakan Keuntungan Toko")
st.write("Gunakan slider untuk menguji skenario 'What-If'.")

# --- SIDEBAR: Variabel Kontrol ---
st.sidebar.header("Tuas Kebijakan (Intervensi)")
iklan_slider = st.sidebar.slider("Anggaran Iklan (Juta)", 0, 50, 10)
diskon_slider = st.sidebar.slider("Besaran Diskon (%)", 0, 50, 10)

# --- ENGINE: Jalankan Simulasi ---
hasil_pred, delta = run_simulation(iklan_slider, diskon_slider)

# --- UI: Tampilkan Hasil ---
col1, col2 = st.columns(2)
col1.metric("Prediksi Keuntungan", f"Rp {hasil_pred:.2f} Jt", f"{delta:.2f} Jt")
col2.write(f"Skenario ini menghasilkan perubahan sebesar {delta:.2f} Juta dibandingkan kondisi baseline.")

# Visualisasi Perbandingan
data_plot = pd.DataFrame({
    'Skenario': ['Baseline', 'Intervensi'],
    'Keuntungan': [baseline_pred, hasil_pred]
})
st.bar_chart(data=data_plot, x='Skenario', y='Keuntungan')

# ============================================================
# Langkah 5: Tampilkan Statistik di Streamlit
# ============================================================
with st.expander("📊 Statistik Deskriptif & Evaluasi Model"):
    st.subheader("Ringkasan Data Historis")
    st.dataframe(df)

    st.subheader("Matriks Korelasi")
    st.dataframe(corr_matrix.style.format("{:.4f}"))

    st.subheader("Uji Signifikansi Korelasi")
    corr_results = []
    for col in ['Iklan (Juta)', 'Diskon (%)']:
        r, p_val = stats.pearsonr(df[col], df['Keuntungan (Juta)'])
        corr_results.append({'Fitur': col, 'r': f"{r:+.4f}", 'p-value': f"{p_val:.4f}"})
    st.dataframe(pd.DataFrame(corr_results))

    st.subheader("Metrik Evaluasi Model")
    metrics_df = pd.DataFrame({
        'Metrik': ['R²', 'MAE (Juta)', 'RMSE (Juta)', 'MAPE (%)'],
        'Nilai': [f"{r2:.4f}", f"{mae:.2f}", f"{rmse:.2f}", f"{mape:.2f}"]
    })
    st.dataframe(metrics_df)

    st.subheader("Koefisien Model")
    coef_df = pd.DataFrame({
        'Variabel': ['Intercept', 'Iklan', 'Diskon'],
        'Koefisien': [f"{model.intercept_:.4f}", f"{model.coef_[0]:.4f}", f"{model.coef_[1]:.4f}"]
    })
    st.dataframe(coef_df)

    # Scatter plot data aktual vs prediksi
    st.subheader("Aktual vs Prediksi")
    pred_actual_df = pd.DataFrame({
        'Aktual': y_train,
        'Prediksi': y_pred_train
    })
    st.line_chart(pred_actual_df)
