import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from transformers import pipeline

# Download stopwords
nltk.download('stopwords')

# Load ML model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Load BERT model
@st.cache_resource
def load_bert():
    return pipeline("sentiment-analysis")

bert_model = load_bert()

# Stopwords
stop_words = set(stopwords.words('english'))

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# UI
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

st.title("🛒 Sentiment Analyzer (ML + BERT)")
st.markdown("### ✨ Compare Machine Learning vs Deep Learning")

# Model selection
model_choice = st.selectbox("Choose Model", ["ML Model (Fast)", "BERT Model (Advanced)"])

user_input = st.text_area("✍️ Enter your review:")

if st.button("Analyze Sentiment"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review")
    else:

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

        else:
            result = bert_model(user_input)[0]

            label = result['label']
            score = result['score']

            st.subheader("🤖 BERT Result")

            if label == "POSITIVE":
                st.success("😊 Positive (BERT)")
            else:
                st.error("😡 Negative (BERT)")

            st.write("Confidence:", round(score, 2))
