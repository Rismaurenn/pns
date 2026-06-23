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

# Menyiapkan data historis sederhana
# Fitur: [Iklan (Juta), Diskon (%)]
X_train = np.array([[5, 10], [10, 20], [15, 5], [20, 25], [25, 15]])
# Target: Keuntungan (Juta)
y_train = np.array([50, 80, 110, 90, 150])

# Melatih model (Mesin Replika)
model = LinearRegression().fit(X_train, y_train)

# Menetapkan Skenario Dasar (Baseline)
# Kondisi saat ini: Iklan 10 Juta, Diskon 10%
baseline_input = np.array([[10, 10]])
baseline_pred = model.predict(baseline_input)[0]
print(f"Prediksi Keuntungan Baseline: Rp {baseline_pred:.2f} Juta")

# ============================================================
# Langkah 2: Membangun Logika Simulator (Analisis What-If)
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
# Langkah 3: Implementasi UI Interaktif (Konsep Streamlit)
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
