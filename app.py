import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Konfigurasi halaman utama biar tampilannya rapi dan responsif
st.set_page_config(page_title="Sistem Deteksi & Prediksi Kesehatan", layout="centered")

# ==========================================
# LOAD MODEL & SCALER (Hasil Soal 8)
# ==========================================
@st.cache_resource
def load_artifacts():
    with open('model_naive_bayes_smote.pkl', 'rb') as f:
        model_nb = pickle.load(f)
    with open('scaler_nb.pkl', 'rb') as f:
        scaler_nb = pickle.load(f)

    try:
        with open('model_linear_regression.pkl', 'rb') as f:
            model_lr = pickle.load(f)
        with open('scaler_lr.pkl', 'rb') as f:
            scaler_lr = pickle.load(f)
    except:
        model_lr = None
        scaler_lr = None

    return model_nb, scaler_nb, model_lr, scaler_lr

model_nb, scaler_nb, model_lr, scaler_lr = load_artifacts()

# ==========================================
# 2.) NAVIGASI SIDEBAR (Halaman A & B)
# ==========================================
st.sidebar.markdown("## 🧭 Navigasi Menu")
pilihan_halaman = st.sidebar.radio(
    "Pilih Fitur Aplikasi:",
    ["Prediksi Glucose (Linear Regression)", "Klasifikasi Outcome (Naive Bayes)"]
)

# Struktur 16 fitur bawaan dataset awal biar urutannya konsisten dengan proses training
nama_fitur = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age", "Fitur_Tambahan_1", "Fitur_Tambahan_2",
    "Fitur_Tambahan_3", "Fitur_Tambahan_4", "Fitur_Tambahan_5", "Fitur_Tambahan_6",
    "Fitur_Tambahan_7", "Fitur_Tambahan_8"
]

# ----------------------------------------------------------------------------------
# 1.) Halaman a: Prediksi Glucose (Linear Regression)
# ----------------------------------------------------------------------------------
if pilihan_halaman == "Prediksi Glucose (Linear Regression)":
    st.title("📊 Estimasi Kadar Glucose Pasien")
    st.write("Halaman ini memanfaatkan model *Linear Regression* untuk memprediksi kadar Glucose berdasarkan parameter klinis pendukung.")

    if model_lr is None:
        st.warning("⚠️ Berkas 'model_linear_regression.pkl' tidak ditemukan di sistem. Pastikan proses ekspor di Soal 8 berjalan lancar.")
    else:
        st.subheader("📝 Form Input Data Klinis (Urutan Sesuai Dataset):")

        # User menginput 15 fitur pendukung
        fitur_lr = [f for f in nama_fitur if f != 'Glucose']
        inputs_lr = {}

        col1, col2 = st.columns(2)
        for i, feat in enumerate(fitur_lr):
            with col1 if i % 2 == 0 else col2:
                val = st.number_input(f"Masukkan {feat}:", min_value=0.0, value=10.0, step=1.0, key=f"lr_{feat}")
                inputs_lr[feat] = val

        st.markdown("---")
        if st.button("Hitung Estimasi Glucose", type="primary"):
            # SOLUSI: Kita susun ulang jadi 16 fitur utuh sesuai urutan asli training.
            # Nilai Glucose diisi dummy 0.0 dulu karena dia adalah target yang mau dicari.
            full_inputs_lr = []
            for feat in nama_fitur:
                if feat == 'Glucose':
                    full_inputs_lr.append(0.0) # Dummy value agar total tetap 16 fitur
                else:
                    full_inputs_lr.append(inputs_lr[feat])

            input_array = np.array([full_inputs_lr])
            input_scaled = scaler_lr.transform(input_array)
            hasil_prediksi = model_lr.predict(input_scaled)
            st.success(f"📈 Hasil Prediksi Kadar Glucose Pasien: **{hasil_prediksi[0]:.2f}**")

# ----------------------------------------------------------------------------------
# 2.) Halaman b: Klasifikasi Outcome (Naive Bayes)
# ----------------------------------------------------------------------------------
elif pilihan_halaman == "Klasifikasi Outcome (Naive Bayes)":
    st.title("🩺 Diagnosis Risiko Diabetes")
    st.write("Halaman ini menggunakan algoritma *Gaussian Naive Bayes* yang telah dioptimalkan dengan SMOTE untuk mengklasifikasikan status risiko pasien.")

    st.subheader("📝 Form Input Parameter Klinis Lengkap (16 Fitur Berurutan):")
    inputs_nb = []

    col1, col2 = st.columns(2)
    for i, feat in enumerate(nama_fitur):
        with col1 if i % 2 == 0 else col2:
            val = st.number_input(f"Nilai {feat}:", min_value=0.0, value=0.0, step=0.1, key=f"nb_{feat}")
            inputs_nb.append(val)

    st.markdown("---")
    if st.button("Mulai Analisis Diagnosis", type="primary"):
        input_array = np.array([inputs_nb])
        input_scaled = scaler_nb.transform(input_array)
        hasil_klasifikasi = model_nb.predict(input_scaled)

        st.subheader("🔍 Hasil Kesimpulan Sistem:")
        if hasil_klasifikasi[0] == 1:
            st.error("⚠️ Hasil Analisis: TERINDIKASI RISIKO DIABETES (Positif / Kelas 1)")
        else:
            st.success("✅ Hasil Analisis: PASIEN DINYATAKAN AMAN / NEGATIF (Negatif / Kelas 0)")
