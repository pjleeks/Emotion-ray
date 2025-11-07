# ======================================
# 🧠 Emotion-ray: Streamlit App
# ======================================

import streamlit as st
import joblib

# --- Load model and vectorizer ---
@st.cache_resource
def load_model():
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    model = joblib.load("svm_emotion_classifier.pkl")
    return vectorizer, model

vec, model = load_model()

# --- App title ---
st.title("🧠 Emotion-ray: AI Emotion Classifier")
st.markdown("Analyze the emotional tone of text using your trained SVM model.")

# --- User input ---
user_input = st.text_area("Enter some text to analyze:")

# --- Prediction ---
if st.button("Detect Emotion"):
    if user_input.strip():
        # Transform input text
        X = vec.transform([user_input])
        pred = model.predict(X)[0]

        # Display result
        st.success(f"**Predicted Emotion:** {pred}")
    else:
        st.warning("Please enter some text first.")

# --- Footer ---
st.markdown("---")
st.caption("Model: SVM + TF-IDF | Project: Emotion-ray | Built for emotion-aware automation 🌈")
