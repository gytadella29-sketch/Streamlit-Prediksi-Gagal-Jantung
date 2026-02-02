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
st.write(
    "Aplikasi ini memprediksi **risiko penyakit jantung** "
    "menggunakan model **Stacking Ensemble Machine Learning**."
)

st.markdown("---")

# ===============================
# INPUT DATA PASIEN
# ===============================
st.subheader("📝 Input Data Pasien")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Umur", 20, 100, 50)
    resting_bp = st.number_input("Tekanan Darah Istirahat", 0, 200, 120)
    cholesterol = st.number_input("Kolesterol", 0, 600, 200)
    max_hr = st.number_input("Max Heart Rate", 60, 220, 150)

with col2:
    sex = st.selectbox("Jenis Kelamin", ["M", "F"])
    fasting_bs = st.selectbox("Gula Darah Puasa > 120 mg/dl", [0, 1])
    exercise_angina = st.selectbox("Angina saat Olahraga", ["Y", "N"])
    oldpeak = st.number_input("Oldpeak", -2.6, 7.0, 1.0)

st.markdown("---")

# ===============================
# PREDIKSI
# ===============================
if st.button("🔍 Prediksi Risiko"):

    # ===============================
    # DATA INPUT (SAMA DENGAN TRAINING)
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

    # Pastikan urutan kolom IDENTIK
    df = df.reindex(columns=columns)

    # ===============================
    # SCALING & PREDIKSI
    # ===============================
    df_scaled = scaler.transform(df)
    prob = model.predict_proba(df_scaled)[0][1]

    # Threshold medis (lebih stabil)
    THRESHOLD = 0.60

    # Confidence
    confidence = abs(prob - 0.5) * 2

    if confidence >= 0.7:
        conf_label = "Sangat Tinggi"
    elif confidence >= 0.5:
        conf_label = "Tinggi"
    elif confidence >= 0.3:
        conf_label = "Sedang"
    else:
        conf_label = "Rendah"

    # ===============================
    # OUTPUT
    # ===============================
    st.markdown("## 📊 Hasil Prediksi")

    if prob >= THRESHOLD:
        st.error("⚠️ **RISIKO TINGGI PENYAKIT JANTUNG**")
    else:
        st.success("✅ **RISIKO RENDAH PENYAKIT JANTUNG**")

    st.markdown("### 📈 Probabilitas Risiko")
    st.write(f"**{prob:.2%}**")
    st.progress(int(prob * 100))

    st.markdown("### 🧠 Confidence Model")
    st.write(f"**{confidence:.2%}**")
    st.write(f"Kategori Confidence: **{conf_label}**")

    st.markdown("### 📝 Interpretasi")
    if prob >= THRESHOLD:
        st.write(
            "Model memprediksi **risiko tinggi penyakit jantung**. "
            "Disarankan melakukan pemeriksaan medis lanjutan."
        )
    else:
        st.write(
            "Model memprediksi **risiko rendah penyakit jantung**. "
            "Tetap jaga pola hidup sehat."
        )

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("Sistem Prediksi Penyakit Jantung")
