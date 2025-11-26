import streamlit as st
import joblib
import urllib.request
import os
import numpy as np

# -------------------------------------------------------
# Page settings
# -------------------------------------------------------
st.set_page_config(page_title="Spam Mail Detector", page_icon="📧", layout="centered")

MODEL_URL = "https://github.com/tanjinatabassum/spam-mail-predictor/releases/download/v1.0/model_rf.pkl"
MODEL_PATH = "model_rf.pkl"        # local cached file
VECTORIZER_PATH = "vectorizer.pkl" # this stays local


# -------------------------------------------------------
# Download model if not present
# -------------------------------------------------------
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model... This will happen only once."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


# -------------------------------------------------------
# Load model & vectorizer only once
# -------------------------------------------------------
@st.cache_resource
def load_model_and_vectorizer():
    # Download model from GitHub Release if needed
    model_path = download_model()

    # Load model & vectorizer
    model = joblib.load(model_path)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


model, vectorizer = load_model_and_vectorizer()

# -------------------------------------------------------
# UI
# -------------------------------------------------------
st.title("📧 Spam Mail Detection")
st.write("Paste any email text below to classify it as **Spam** or **Not Spam**.")

email_text = st.text_area("Email Content", height=250, placeholder="Type or paste an email message here...")


# -------------------------------------------------------
# Prediction logic
# -------------------------------------------------------
if st.button("Predict"):
    if not email_text.strip():
        st.warning("Please enter an email message first.")
    else:
        X = vectorizer.transform([email_text])

        # Get raw prediction
        pred_raw = model.predict(X)[0]
        classes = model.classes_
        proba = model.predict_proba(X)[0]

        # Detect which class is spam
        if set(classes) == {0, 1}:
            spam_class = 1
            not_spam_class = 0
        elif "spam" in classes and "ham" in classes:
            spam_class = "spam"
            not_spam_class = "ham"
        else:
            spam_class = classes[1]
            not_spam_class = classes[0]

        # Correct probability mapping
        spam_prob = proba[list(classes).index(spam_class)]
        not_spam_prob = proba[list(classes).index(not_spam_class)]

        # Final human-readable output
        if pred_raw == spam_class:
            st.metric(
                label="Prediction",
                value="🚫 Spam",
                delta=f"{spam_prob * 100:.2f}% chance of Spam"
            )
        else:
            st.metric(
                label="Prediction",
                value="✔️ Not Spam",
                delta=f"{not_spam_prob * 100:.2f}% chance of Not Spam"
            )
