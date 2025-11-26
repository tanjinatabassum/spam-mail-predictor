import streamlit as st
import joblib
import numpy as np

# -------------------------------------------------------
# Page settings
# -------------------------------------------------------
st.set_page_config(page_title="Spam Mail Detector", page_icon="📧", layout="centered")

# -------------------------------------------------------
# Load model & vectorizer only once
# -------------------------------------------------------
@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load("model_rf.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
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
        # Transform text
        X = vectorizer.transform([email_text])

        # Raw prediction
        pred_raw = model.predict(X)[0]
        classes = model.classes_
        proba = model.predict_proba(X)[0]

        # Detect spam class
        if set(classes) == {0, 1}:
            spam_class = 1
            not_spam_class = 0
        elif "spam" in classes and "ham" in classes:
            spam_class = "spam"
            not_spam_class = "ham"
        else:
            spam_class = classes[1]
            not_spam_class = classes[0]

        spam_idx = list(classes).index(spam_class)
        not_spam_idx = list(classes).index(not_spam_class)

        spam_prob = proba[spam_idx]
        not_spam_prob = proba[not_spam_idx]

        # -------------------------------------------------------
        # Final output to user
        # -------------------------------------------------------
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
