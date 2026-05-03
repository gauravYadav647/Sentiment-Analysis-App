import streamlit as st
import pickle
import re
from nltk.corpus import stopwords

# Page settings
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Load stopwords
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# UI
st.title("🛒 Amazon Review Sentiment Analyzer")
st.markdown("### ✨ Analyze product reviews using Machine Learning")
st.write("Enter your review below and get instant sentiment prediction.")

# Input box
user_input = st.text_area("✍️ Enter your review here:")

# Button action
if st.button("🔍 Analyze Sentiment"):

    # Check empty input
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review first")

    else:
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])

        result = model.predict(vector)[0]

        prob = model.predict_proba(vector)
        confidence = max(prob[0])

        st.subheader("📊 Result")

        if result == "Positive":
            st.success("😊 Positive Review")
        elif result == "Negative":
            st.error("😡 Negative Review")
        else:
            st.warning("😐 Neutral Review")

        st.write("Confidence Score:", round(confidence, 2))

        # Show cleaned text (for understanding)
        st.markdown("### 🧹 Cleaned Text")
        st.write(cleaned)