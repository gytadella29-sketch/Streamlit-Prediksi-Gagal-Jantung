import streamlit as st
import pandas as pd
import joblib

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Prediksi Penyakit Jantung",
    page_icon="❤️",
    layout="wide"
)

# ===============================
# LOAD MODEL & ARTIFACT
# ===============================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# ===============================
# JUDUL
# ===============================
st.title("❤️ Sistem Prediksi Penyakit Jantung")
st.markdown(
    """
    Aplikasi ini digunakan untuk **memprediksi tingkat risiko penyakit jantung**
    menggunakan model **Stacking Ensemble Machine Learning**.
    """
)

st.markdown("---")

# ===============================
# INPUT DATA PASIEN
# ===============================
st.subheader("📝 Input Data Pasien")

col1, col2 = st.columns(2)

with col1:
    age_input = st.text_input(
        "Umur",
        placeholder="Masukkan umur pasien (contoh: 45)"
    )

    resting_bp = st.number_input(
        "Tekanan Darah Istirahat (mmHg)",
        0, 300, 120
    )

    cholesterol = st.number_input(
        "Kolesterol (mg/dL)",
        0, 700, 200
    )

    max_hr = st.number_input(
        "Max Heart Rate",
        0, 250, 150
    )

with col2:
    sex = st.selectbox("Jenis Kelamin", ["M", "F"])
    fasting_bs = st.selectbox(
        "Gula Darah Puasa > 120 mg/dL",
        [0, 1]
    )
    exercise_angina = st.selectbox(
        "Angina saat Olahraga",
        ["Y", "N"]
    )
    oldpeak = st.number_input(
        "Oldpeak",
        -5.0, 10.0, 1.0
    )

st.markdown("---")

# ===============================
# BUTTON
# ===============================
col_pred, col_reset = st.columns(2)

with col_pred:
    predict_btn = st.button("🔍 Prediksi Risiko")

with col_reset:
    reset_btn = st.button("🔄 Reset Data")

if reset_btn:
    st.experimental_rerun()

# ===============================
# PREDIKSI
# ===============================
if predict_btn:

    # ===== VALIDASI UMUR =====
    if age_input.strip() == "":
        st.warning("⚠️ Umur wajib diisi.")
        st.stop()

    try:
        age = int(age_input)
        if age < 0 or age > 120:
            st.warning("⚠️ Umur harus berada di antara 0 – 120 tahun.")
            st.stop()
    except ValueError:
        st.warning("⚠️ Umur harus berupa angka.")
        st.stop()

    # ===============================
    # DATA INPUT
    # ===============================
    input_data = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "MaxHR": max_hr,
        "FastingBS": fasting_bs,
        "Oldpeak": oldpeak,
        "Sex_M": 1 if sex == "M" else 0,
        "ExerciseAngina_Y": 1 if exercise_angina == "Y" else 0
    }

    df = pd.DataFrame([input_data])
    df = df.reindex(columns=columns)

    # ===============================
    # SCALING & PREDIKSI
    # ===============================
    df_scaled = scaler.transform(df)
    prob = model.predict_proba(df_scaled)[0][1]

    # ===============================
    # KATEGORI RISIKO
    # ===============================
    if prob < 0.4:
        risk_label = "RENDAH"
        color = "green"
    elif prob < 0.7:
        risk_label = "SEDANG"
        color = "orange"
    else:
        risk_label = "TINGGI"
        color = "red"

    # Confidence
    confidence = abs(prob - 0.5) * 2

    # ===============================
    # OUTPUT
    # ===============================
    st.markdown("## 📊 Hasil Prediksi")

    if risk_label == "TINGGI":
        st.error("⚠️ **RISIKO TINGGI PENYAKIT JANTUNG**")
    elif risk_label == "SEDANG":
        st.warning("⚠️ **RISIKO SEDANG PENYAKIT JANTUNG**")
    else:
        st.success("✅ **RISIKO RENDAH PENYAKIT JANTUNG**")

    st.markdown(f"### 🩺 Tingkat Risiko: **:{color}[{risk_label}]**")

    st.markdown("### 📈 Probabilitas Risiko")
    st.write(f"**{prob:.2%}**")
    st.progress(int(prob * 100))

    st.markdown("### 🧠 Confidence Model")
    st.write(f"**{confidence:.2%}**")

    st.markdown("### 📝 Interpretasi")
    if risk_label == "TINGGI":
        st.write(
            "Pasien memiliki **risiko tinggi penyakit jantung**. "
            "Disarankan segera melakukan pemeriksaan medis lanjutan."
        )
    elif risk_label == "SEDANG":
        st.write(
            "Pasien memiliki **risiko sedang penyakit jantung**. "
            "Perlu perhatian terhadap pola hidup dan pemeriksaan berkala."
        )
    else:
        st.write(
            "Pasien memiliki **risiko rendah penyakit jantung**. "
            "Tetap menjaga gaya hidup sehat."
        )

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("Sistem Prediksi Penyakit Jantung | Machine Learning Ensemble")
