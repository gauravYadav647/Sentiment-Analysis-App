import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Try importing BERT (safe)
try:
    from transformers import pipeline
    BERT_AVAILABLE = True
except:
    BERT_AVAILABLE = False

# Download stopwords
nltk.download('stopwords')

# Load ML model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Load BERT (only if available)
@st.cache_resource
def load_bert():
    if BERT_AVAILABLE:
        return pipeline("sentiment-analysis")
    return None

bert_model = load_bert()

# Stopwords
stop_words = set(stopwords.words('english'))

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# UI Config
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

st.title("🛒 Sentiment Analyzer (ML + BERT)")
st.markdown("### ✨ Analyze product reviews using Machine Learning")

# Sidebar info
st.sidebar.header("ℹ️ About")
st.sidebar.write("Fast Model: TF-IDF + Logistic Regression")
st.sidebar.write("Advanced Model: BERT (context-aware but slower)")

# Model selection
model_choice = st.selectbox(
    "Choose Model",
    ["ML Model (Fast)", "BERT Model (Advanced)"]
)

# Input
user_input = st.text_area("✍️ Enter your review:")

# Button
if st.button("🔍 Analyze Sentiment"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review")
    else:

        # ---------------- ML MODEL ----------------
        if model_choice == "ML Model (Fast)":
            cleaned = clean_text(user_input)
            vector = vectorizer.transform([cleaned])

            result = model.predict(vector)[0]
            prob = model.predict_proba(vector)
            confidence = max(prob[0])

            st.subheader("⚡ ML Result")

            if result == "Positive":
                st.success("😊 Positive Review")
            else:
                st.error("😡 Negative Review")

            st.write("Confidence:", round(confidence, 2))

            st.markdown("### 🧹 Cleaned Text")
            st.write(cleaned)

        # ---------------- BERT MODEL ----------------
        else:
            if bert_model is not None:
                try:
                    result = bert_model(user_input)[0]
                    label = result['label']
                    score = result['score']

                    st.subheader("🤖 BERT Result")

                    if label == "POSITIVE":
                        st.success("😊 Positive (BERT)")
                    else:
                        st.error("😡 Negative (BERT)")

                    st.write("Confidence:", round(score, 2))

                except Exception:
                    st.warning("⚠️ BERT model failed, please try ML model")

            else:
                st.warning("⚠️ BERT not available. Please use ML model.")

# Footer note
st.caption("Note: ML model is faster. BERT is more accurate but slower.")
