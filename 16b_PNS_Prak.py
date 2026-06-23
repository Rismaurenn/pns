"""
Lecture Note Praktikum: Pemodelan dan Simulasi
Minggu 16: UAS - Integrasi Akhir dan Strategi Demonstrasi Produk

=== ALUR INTEGRASI (Pipeline) ===
1. Slider (Intervensi Pengguna)
   → 2. Scaler Transform
   → 3. Model ML Predict (Inference)
   → 4. Matriks Keputusan (ML + Statis)
   → 5. SAW Calculation → Ranking
   → 6. SHAP Explainability (XAI)
   → 7. What-If Scenario Analysis
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib

# ============================================================
# Setup Paths (relatif terhadap folder app/)
# ============================================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)                          # naik ke tugas praktikum/
MODEL_DIR = os.path.join(BASE_DIR, 'output', 'minggu15')     # Model dari M15
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'minggu16')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 0. Load / Train Model (MLOps: Model Serialization)
# ============================================================
model_path = os.path.join(MODEL_DIR, 'model_risiko_v1.joblib')
scaler_path = os.path.join(MODEL_DIR, 'scaler_risiko_v1.joblib')

if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    model_loaded = True
else:
    # Fallback: latih model sederhana jika file belum ada
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    X_train = np.array([[60, 2], [70, 4], [80, 6], [90, 8], [100, 10]])
    y_train = np.array([10, 25, 45, 70, 95])
    scaler = StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)
    model = LinearRegression().fit(X_scaled, y_train)
    model_loaded = False

# ============================================================
# Fungsi SAW (Simple Additive Weighting)
# ============================================================
def saw_hybrid(X, jenis_kriteria, W):
    """Normalisasi dan pembobotan SAW dengan benefit/cost."""
    R = np.zeros(X.shape)
    for j in range(X.shape[1]):
        if jenis_kriteria[j] == 1:      # Benefit
            R[:, j] = X[:, j] / np.max(X[:, j])
        else:                           # Cost
            R[:, j] = np.min(X[:, j]) / X[:, j]
    return np.dot(R, W)

# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="UAS - Simulator Risiko Mesin", layout="wide")
st.title("🛠️ UAS: Simulator Risiko Mesin Terintegrasi")
st.markdown("**Integrasi:** *ML Prediction → SPK SAW → XAI (SHAP) → What-If*")

# ----------------------------------------------------------
# SIDEBAR: Variabel Kontrol (Intervensi Pengguna)
# ----------------------------------------------------------
st.sidebar.header("🎛️ Tuas Intervensi (What-If)")
suhu = st.sidebar.slider("🌡️ Suhu Mesin (°C)", 0, 150, 85)
getaran = st.sidebar.slider("📳 Getaran Mesin (mm/s)", 0, 20, 7)

# ----------------------------------------------------------
# 1 & 2. Scaler Transform → ML Prediction (Inference)
# ----------------------------------------------------------
input_data = np.array([[suhu, getaran]])
input_scaled = scaler.transform(input_data)
prediksi_risiko = model.predict(input_scaled)[0]

# ----------------------------------------------------------
# 3. Matriks Keputusan (Hasil ML + Kolom Statis)
# ----------------------------------------------------------
# Alternatif A: risiko dari ML; B dan C: nilai tetap
matriks_x = np.array([
    [prediksi_risiko, 5, 80],   # Alternatif A (dipengaruhi slider)
    [30.0,            8, 95],   # Alternatif B
    [70.0,            3, 70]    # Alternatif C
])

alternatif_labels = ['Alternatif_A', 'Alternatif_B', 'Alternatif_C']
kriteria_labels  = ['Risiko_ML (Cost)', 'Biaya (Cost)', 'Efisiensi (Benefit)']

bobot_ahp = np.array([0.5, 0.3, 0.2])   # Bobot prioritas dari AHP
jenis     = [0, 0, 1]                    # 0 = Cost, 1 = Benefit

# ----------------------------------------------------------
# 4. SAW Calculation → Ranking
# ----------------------------------------------------------
skor_akhir = saw_hybrid(matriks_x, jenis, bobot_ahp)
df_ranking = pd.DataFrame({'Skor': skor_akhir}, index=alternatif_labels)
df_ranking = df_ranking.sort_values(by='Skor', ascending=False)

# ============================================================
# MAIN PANEL: Tampilan Hasil
# ============================================================

# --- Row 1: Metric Cards ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("🔮 Risiko (ML)", f"{prediksi_risiko:.2f}")
col2.metric("🥇 Alternatif Terbaik", df_ranking.index[0].replace('_', ' '))
col3.metric("📈 Skor Tertinggi", f"{df_ranking['Skor'].iloc[0]:.3f}")
col4.metric("⚙️ Suhu / Getaran", f"{suhu}°C / {getaran} mm/s")

if not model_loaded:
    st.info("ℹ️  Model .joblib tidak ditemukan — menggunakan model darurat (fallback). Jalankan 15b_PNS_Prak.py untuk menyimpan model permanen.")

# --- Row 2: Decision Matrix ---
st.subheader("📋 Matriks Keputusan")
df_matriks = pd.DataFrame(matriks_x, columns=kriteria_labels, index=alternatif_labels)

# Warna baris Alternatif A (yang dipengaruhi slider)
def style_matriks(df):
    styles = pd.DataFrame('', index=df.index, columns=df.columns)
    styles.loc['Alternatif_A', :] = 'background-color: #fff3cd'
    return styles

st.dataframe(df_matriks.style.apply(style_matriks, axis=None)
             .highlight_max(axis=0, subset=(slice(None), ['Efisiensi (Benefit)']), color='#d4edda')
             .highlight_min(axis=0, subset=(slice(None), ['Risiko_ML (Cost)', 'Biaya (Cost)']), color='#d4edda'))

# --- Row 3: SHAP Explainability (XAI) ---
st.subheader("🔍 Mengapa Hasilnya Demikian?")
st.markdown("*Penjelasan kontribusi fitur terhadap prediksi risiko menggunakan SHAP.*")

try:
    import shap
    # Explainer untuk model LinearRegression
    explainer = shap.Explainer(model, input_scaled)
    shap_values = explainer(input_scaled)

    fig_shap, ax_shap = plt.subplots(figsize=(7, 3))
    shap.waterfall_plot(shap_values[0], max_display=3, show=False)
    st.pyplot(plt.gcf())
    plt.close()
except ImportError:
    st.info("📦 SHAP tidak terinstall. Untuk melihat penjelasan fitur, jalankan:  `pip install shap`")
    # Tampilkan koefisien model sebagai alternatif
    st.markdown("**Koefisien Model (sebagai alternatif penjelasan):**")
    coef_df = pd.DataFrame({
        'Fitur': ['Suhu_Mesin', 'Getaran_Mesin'],
        'Nilai Input': [suhu, getaran],
        'Koefisien': model.coef_,
        'Kontribusi': model.coef_ * input_scaled[0]
    })
    st.dataframe(coef_df)
except Exception as e:
    st.warning(f"SHAP plot gagal: {e}")
    coef_df = pd.DataFrame({
        'Fitur': ['Suhu_Mesin', 'Getaran_Mesin'],
        'Koefisien': model.coef_
    })
    st.dataframe(coef_df)

# --- Row 4: SAW Ranking Bar Chart ---
st.subheader("📊 Ranking Keputusan Akhir (Metode SAW)")

fig, ax = plt.subplots(figsize=(9, 4.5))
bar_colors = ['#e74c3c' if df_ranking.index[i] == 'Alternatif_A'
              else '#2ecc71' if df_ranking.index[i] == 'Alternatif_B'
              else '#f39c12'
              for i in range(len(df_ranking))]
bars = ax.bar(df_ranking.index, df_ranking['Skor'], color=bar_colors,
              edgecolor='gray', linewidth=1.2, width=0.55)

mean_skor = df_ranking['Skor'].mean()
ax.axhline(y=mean_skor, color='blue', linestyle='--', linewidth=1.5,
           label=f"Rata-rata ({mean_skor:.3f})")
ax.set_ylabel("Skor Preferensi", fontsize=11)
ax.set_ylim(0, 1.0)
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.2, axis='y')
ax.set_title("Perbandingan Skor Akhir Setiap Alternatif", fontsize=13)

for bar, skor in zip(bars, df_ranking['Skor']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{skor:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
st.pyplot(fig)

# Simpan ke folder dokumentasi
plt.savefig(os.path.join(OUTPUT_DIR, '01_ranking_keputusan.png'), dpi=150, bbox_inches='tight')
plt.close()
st.caption(f"📸 Plot otomatis tersimpan di: `{os.path.join(OUTPUT_DIR, '01_ranking_keputusan.png')}`")

# --- Row 5: What-If Scenario Analysis ---
st.subheader("🔄 Analisis Skenario What-If")
st.markdown("""
**Demonstrasi:** *"Jika suhu mesin turun 20%, bagaimana perubahan risiko dan ranking?"*
""")

suhu_baru = suhu * 0.8                     # Turun 20%
input_baru = np.array([[suhu_baru, getaran]])
input_baru_scaled = scaler.transform(input_baru)
prediksi_baru = model.predict(input_baru_scaled)[0]

col_a, col_b = st.columns(2)
with col_a:
    st.metric("📊 Skenario Saat Ini",
              f"Suhu: {suhu}°C | Getaran: {getaran} mm/s",
              delta=f"Risiko: {prediksi_risiko:.1f}",
              delta_color="off")
with col_b:
    st.metric("📊 Skenario: Suhu Turun 20%",
              f"Suhu: {suhu_baru:.0f}°C | Getaran: {getaran} mm/s",
              delta=f"Risiko: {prediksi_baru:.1f}",
              delta_color="off")

penurunan = prediksi_risiko - prediksi_baru
persen_penurunan = (penurunan / prediksi_risiko * 100) if prediksi_risiko != 0 else 0

if penurunan > 0:
    st.success(f"💡 **Penurunan Risiko:** {penurunan:.1f} poin ({persen_penurunan:.1f}%) — "
               f"Mengurangi suhu mesin menurunkan risiko kegagalan secara signifikan.")
else:
    st.info("💡 Tidak ada perubahan risiko yang signifikan pada skenario ini.")

# --- Footer / Keterangan ---
st.markdown("---")
st.markdown("""
**📌 Catatan Presentasi UAS:**
1. **Context:** Masalah nyata — risiko kegagalan mesin industri.
2. **Live Interaction:** Geser slider *Suhu* dan *Getaran* untuk melihat perubahan prediksi & ranking secara *real-time*.
3. **Analysis:** Perhatikan bagaimana perubahan input menggeser nilai pada baris **Alternatif_A** di Matriks Keputusan.
4. **Justification:** Gunakan grafik **SHAP** untuk menjelaskan *mengapa* model memberikan prediksi tersebut.
""")
