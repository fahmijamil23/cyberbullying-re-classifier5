
import streamlit as st

st.set_page_config(
    page_title="Cyberbullying Classifier",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Cyberbullying Tweet Classifier")
st.markdown("**Final Project — Group 3 (Baby Python) | Data Science Batch 59 Digital Skola**")
st.markdown("---")

st.markdown("""
### Selamat Datang!

Aplikasi ini mengklasifikasikan tweet apakah mengandung unsur **cyberbullying** atau tidak,
menggunakan metode **Hierarchical Classification** dengan TF-IDF + N-Gram, dan model
yang dilatih **terpisah untuk bahasa Indonesia dan Inggris**.

#### 📌 Navigasi:
- 📊 **EDA** — Eksplorasi dan visualisasi dataset
- 🔍 **Classifier** — Deteksi cyberbullying pada tweet (auto-detect bahasa)

> Gunakan menu di **sidebar kiri** untuk berpindah halaman.
""")

st.info("💡 Tip: Klik halaman **Classifier** untuk langsung mencoba prediksi tweet!")
