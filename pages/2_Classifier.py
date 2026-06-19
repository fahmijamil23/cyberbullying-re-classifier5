
import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from contractions import fix
from langdetect import detect, LangDetectException
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

st.set_page_config(page_title="Classifier", page_icon="🔍", layout="centered")

# ── Load 7 file model ──────────────────────────────────────
@st.cache_resource
def load_models():
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("model_stage1_id.pkl", "rb") as f:
        model_stage1_id = pickle.load(f)
    with open("model_stage2_id.pkl", "rb") as f:
        model_stage2_id = pickle.load(f)
    with open("encoder_stage2_id.pkl", "rb") as f:
        encoder_stage2_id = pickle.load(f)
    with open("model_stage1_en.pkl", "rb") as f:
        model_stage1_en = pickle.load(f)
    with open("model_stage2_en.pkl", "rb") as f:
        model_stage2_en = pickle.load(f)
    with open("encoder_stage2_en.pkl", "rb") as f:
        encoder_stage2_en = pickle.load(f)
    return (vectorizer, model_stage1_id, model_stage2_id, encoder_stage2_id,
            model_stage1_en, model_stage2_en, encoder_stage2_en)

(vectorizer, model_stage1_id, model_stage2_id, encoder_stage2_id,
 model_stage1_en, model_stage2_en, encoder_stage2_en) = load_models()

# ── Setup stopword, lemmatization, fix ─────────────────────
stemmer_id = StemmerFactory().create_stemmer()
lemmatizer_en = WordNetLemmatizer()
stop_wordsENCust = set(stopwords.words('english'))
stop_wordsID = set(stopwords.words('indonesian'))

slang_dict = {
    "yg": "yang", "dg": "dengan", "tdk": "tidak", "gak": "tidak",
    "gk": "tidak", "ga": "tidak", "nggak": "tidak", "ngga": "tidak",
    "tak": "tidak", "tp": "tapi", "tpi": "tapi", "krn": "karena",
    "krna": "karena", "karna": "karena", "krena": "karena",
    "utk": "untuk", "u": "untuk", "buat": "untuk", "spy": "supaya",
    "biar": "supaya", "sm": "sama", "ama": "sama", "udh": "sudah",
    "udah": "sudah", "dah": "sudah", "sdh": "sudah", "blm": "belum",
    "blum": "belum", "msh": "masih", "lg": "lagi", "jg": "juga",
    "jga": "juga", "sj": "saja", "aja": "saja", "aj": "saja",
    "jd": "jadi", "jdi": "jadi", "sdg": "sedang", "klo": "kalau",
    "klu": "kalau", "kl": "kalau", "kalo": "kalau", "kpd": "kepada",
    "pd": "pada", "dr": "dari", "dri": "dari", "dgn": "dengan",
    "ttg": "tentang", "gw": "saya", "gue": "saya", "w": "saya",
    "aq": "saya", "ak": "saya", "aku": "saya", "lo": "kamu",
    "lu": "kamu", "elo": "kamu", "loe": "kamu", "km": "kamu",
    "kmu": "kamu", "dy": "dia", "doi": "dia", "mrk": "mereka",
    "kt": "kita", "qt": "kita", "kmi": "kami", "bgt": "banget",
    "bngt": "banget", "bget": "banget", "skrg": "sekarang",
    "skrng": "sekarang", "skg": "sekarang", "trs": "terus",
    "trus": "terus", "jgn": "jangan", "jngn": "jangan",
    "mw": "mau", "mo": "mau", "bs": "bisa", "bsa": "bisa",
    "hrs": "harus", "pls": "tolong", "tlg": "tolong",
    "mksh": "terima kasih", "makasih": "terima kasih",
    "thx": "terima kasih", "tks": "terima kasih",
    "byk": "banyak", "bnyk": "banyak", "dikit": "sedikit",
    "sdikit": "sedikit", "bsr": "besar", "gede": "besar",
    "gde": "besar", "lbh": "lebih", "plg": "paling",
    "emg": "memang", "emng": "memang", "mmg": "memang",
    "knp": "kenapa", "ngp": "kenapa", "knapa": "kenapa",
    "gmn": "bagaimana", "gmna": "bagaimana", "gimana": "bagaimana",
    "bgmn": "bagaimana", "bilang": "berkata", "blg": "berkata",
    "mkn": "makan", "prgi": "pergi", "dtg": "datang",
    "dateng": "datang", "bodo": "bodoh", "bloon": "bodoh",
    "tolol": "bodoh", "bangsat": "bajingan", "bgs": "bagus",
    "jlek": "jelek", "ancur": "hancur", "hancurin": "menghancurkan",
}

def normalize_slang(text):
    if not isinstance(text, str):
        return text
    tokens = text.split()
    normalized = [slang_dict.get(word.lower(), word) for word in tokens]
    return ' '.join(normalized)

def clean_text(text):
    if text is None or (isinstance(text, float)):
        return ""
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'_', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text

def detect_language_safe(text):
    s = str(text).strip()
    if not s:
        return None
    if len(re.sub(r'[^A-Za-z]+', '', s)) < 3:
        return None
    try:
        return detect(s)
    except LangDetectException:
        return None
    except Exception:
        return None

def preprocess_text(text, lang):
    cleaned = clean_text(text)

    if lang == "id":
        cleaned = normalize_slang(cleaned)

    text_fixed = fix(cleaned)
    tokens = word_tokenize(text_fixed)

    if lang == "id":
        tokens = [stemmer_id.stem(w) for w in tokens if w not in stop_wordsID]
    elif lang == "en":
        words = cleaned.split()
        tokens = [lemmatizer_en.lemmatize(w) for w in words if w not in stop_wordsENCust]
    else:
        tokens = [w for w in tokens if w not in stop_wordsID and w not in stop_wordsENCust]

    return tokens

def predict_hierarchical(text):
    lang = detect_language_safe(text)
    if lang not in ['id', 'en']:
        lang = 'en'  # default fallback untuk bahasa lain / gagal deteksi

    processed_tokens = preprocess_text(text, lang)
    processed_text = ' '.join(processed_tokens)
    vec = vectorizer.transform([processed_text])

    if lang == 'id':
        model_s1, model_s2, enc_s2 = model_stage1_id, model_stage2_id, encoder_stage2_id
    else:
        model_s1, model_s2, enc_s2 = model_stage1_en, model_stage2_en, encoder_stage2_en

    pred_s1 = model_s1.predict(vec)[0]
    if pred_s1 == 0:
        return "not_cyberbullying", None, lang

    pred_s2 = model_s2.predict(vec)[0]
    label = enc_s2.inverse_transform([pred_s2])[0]
    return "cyberbullying", label, lang

# ── UI ─────────────────────────────────────────────────────
st.title("🔍 Cyberbullying Tweet Classifier")
st.markdown("**Final Project — Group 3 (Baby Python) | Data Science Batch 59 Digital Skola**")
st.markdown("---")

st.markdown("### Masukkan teks tweet di bawah ini:")
user_input = st.text_area("", placeholder="Ketik tweet di sini (Bahasa Indonesia / English)...", height=150)

color_map = {
    "age":                "🟠",
    "ethnicity":          "🟣",
    "gender":             "🔵",
    "religion":           "🟡",
    "other_cyberbullying":"🔴"
}

if st.button("🔍 Prediksi"):
    if user_input.strip() == "":
        st.warning("⚠️ Teks tidak boleh kosong!")
    else:
        with st.spinner("Menganalisis tweet..."):
            result, category, detected_lang = predict_hierarchical(user_input)

        st.markdown("---")
        st.markdown("### Hasil Prediksi:")
        st.caption(f"Bahasa terdeteksi: **{'Indonesia' if detected_lang == 'id' else 'English'}**")

        if result == "not_cyberbullying":
            st.success("🟢 **Not Cyberbullying**")
            st.info("✅ Tweet ini tidak mengandung unsur cyberbullying.")
        else:
            emoji = color_map.get(category, "⚪")
            st.error("⚠️ Tweet ini terdeteksi mengandung cyberbullying!")
            st.markdown(f"**Kategori:** {emoji} **{category.replace('_', ' ').title()}**")
