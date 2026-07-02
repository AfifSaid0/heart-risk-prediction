import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================
# LOAD MODEL DAN PREPROCESSING
# ==========================

model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ==========================
# KONFIGURASI HALAMAN
# ==========================

st.set_page_config(
    page_title="Heart Risk Prediction",
    page_icon="❤️",
    layout="centered"
)

# ==========================
# HEADER
# ==========================

st.title("❤️ Heart Risk Prediction System")
st.markdown("""
Aplikasi ini digunakan untuk memprediksi **tingkat risiko penyakit jantung**
menggunakan model **Logistic Regression** yang telah dioptimasi menggunakan
**GridSearchCV**.
""")

st.markdown("---")

# ==========================
# SIDEBAR
# ==========================

st.sidebar.title("ℹ️ Informasi Model")

st.sidebar.success("Model : Logistic Regression")

st.sidebar.write("Hyperparameter Tuning : GridSearchCV")

st.sidebar.write("Deployment : Streamlit")

st.sidebar.write("Dataset : Heart Risk Progression")

# ==========================
# FORM INPUT
# ==========================

st.subheader("📋 Masukkan Data Pasien")

umur = st.number_input(
    "Umur",
    min_value=18,
    max_value=100,
    value=30
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=50.0,
    value=25.0
)

sistolik = st.number_input(
    "Tekanan Darah Sistolik",
    min_value=70,
    max_value=250,
    value=120
)

diastolik = st.number_input(
    "Tekanan Darah Diastolik",
    min_value=40,
    max_value=180,
    value=80
)

kolesterol = st.number_input(
    "Kolesterol",
    min_value=100,
    max_value=400,
    value=180
)

langkah_harian = st.number_input(
    "Langkah Harian",
    min_value=0,
    max_value=30000,
    value=6000
)

tingkat_stres = st.slider(
    "Tingkat Stres",
    1,
    10,
    5
)

aktivitas_fisik = st.slider(
    "Aktivitas Fisik (jam/minggu)",
    0,
    20,
    5
)

jam_tidur = st.slider(
    "Jam Tidur",
    1,
    12,
    7
)

skor_diet = st.slider(
    "Skor Diet",
    1,
    10,
    5
)

skor_risiko = st.slider(
    "Skor Risiko",
    1,
    10,
    5
)

alkohol = st.selectbox(
    "Konsumsi Alkohol",
    ["0", "1"]
)

detak = st.selectbox(
    "Kategori Detak Jantung",
    [
        "Normal",
        "Rendah",
        "Tinggi"
    ]
)

merokok = st.selectbox(
    "Status Merokok",
    [
        "Never",
        "Former/Current"
    ]
)

keluarga = st.selectbox(
    "Riwayat Keluarga",
    [
        "No",
        "Yes"
    ]
)

st.markdown("---")

prediksi = st.button("🔍 Prediksi Risiko")

# ==========================
# PROSES PREDIKSI
# ==========================

if prediksi:

    # -------------------------
    # Dummy Encoding
    # -------------------------

    detak_rendah = 1 if detak == "Rendah" else 0
    detak_tinggi = 1 if detak == "Tinggi" else 0

    merokok_never = 1 if merokok == "Never" else 0

    keluarga_yes = 1 if keluarga == "Yes" else 0

    # -------------------------
    # Susun Data Sesuai Training
    # -------------------------

    data_input = pd.DataFrame([{
        "umur": umur,
        "bmi": bmi,
        "sistolik": sistolik,
        "diastolik": diastolik,
        "kolesterol": kolesterol,
        "langkah_harian": langkah_harian,
        "tingkat_stres": tingkat_stres,
        "aktivitas_fisik": aktivitas_fisik,
        "jam_tidur": jam_tidur,
        "skor_diet": skor_diet,
        "alkohol": int(alkohol),
        "skor_risiko": skor_risiko,
        "detak_jantung_Rendah": detak_rendah,
        "detak_jantung_Tinggi": detak_tinggi,
        "status_merokok_Never": merokok_never,
        "riwayat_keluarga_Yes": keluarga_yes
    }])

    # -------------------------
    # StandardScaler
    # -------------------------

    data_scaled = scaler.transform(data_input)

    # -------------------------
    # PCA
    # -------------------------

    data_pca = pca.transform(data_scaled)

    # -------------------------
    # Prediksi
    # -------------------------

    hasil = model.predict(data_pca)

    hasil_label = label_encoder.inverse_transform(hasil)[0]

        # ==========================
    # HASIL PREDIKSI
    # ==========================

    st.markdown("---")

    st.subheader("📊 Hasil Prediksi")

    # Ubah menjadi string agar aman
    hasil_text = str(hasil_label).lower()

    if "low" in hasil_text or hasil_text == "0":
        st.success("🟢 **Low Risk**")
        st.info(
            """
            Pasien memiliki risiko penyakit jantung yang **rendah**.
            
            Tetap pertahankan pola hidup sehat, olahraga rutin,
            konsumsi makanan bergizi, dan lakukan pemeriksaan kesehatan secara berkala.
            """
        )

    elif "medium" in hasil_text or hasil_text == "1":
        st.warning("🟡 **Medium Risk**")
        st.warning(
            """
            Pasien memiliki risiko penyakit jantung **sedang**.
            
            Disarankan mulai memperbaiki pola makan,
            meningkatkan aktivitas fisik,
            mengurangi stres,
            dan melakukan konsultasi dengan tenaga kesehatan.
            """
        )

    else:
        st.error("🔴 **High Risk**")
        st.error(
            """
            Pasien memiliki risiko penyakit jantung **tinggi**.
            
            Disarankan segera melakukan pemeriksaan lebih lanjut
            ke dokter atau fasilitas kesehatan untuk mendapatkan
            penanganan yang tepat.
            """
        )

    st.markdown("---")

    st.caption(
        "Model : Logistic Regression + GridSearchCV | "
        "Deployment menggunakan Streamlit"
    )