# app.py (Rename to streamlit_app.py if desired, but update content)

import streamlit as st
import sys
import os

# Assuming model_predict.py is in the root directory:
# Add the parent directory to the path if needed for deployment services.
# sys.path.append(os.path.dirname(os.path.abspath(__file__))) 
from model_predict import load_emotion_ray_model, predict_emotion

# --- Load Model (Cached for efficiency) ---
@st.cache_resource
def load_assets():
    return load_emotion_ray_model()

vectorizer, classifier = load_assets()

# --- Streamlit UI ---
st.set_page_config(page_title="Emotion-ray Classifier", layout="centered")
st.title("🧠 Emotion-ray: Real-time Text Emotion Classification")
st.markdown("A highly accurate classifier (0.95 F1) boosted by **Back-Translation Augmentation**.")

if vectorizer and classifier:
    user_input = st.text_area("Enter Text to Analyze:", "I am so happy with the final result!", height=150)
    
    if st.button("Analyze Emotion", type="primary", use_container_width=True):
        if user_input:
            with st.spinner('Analyzing with Augmented SVM...'):
                predicted_emotion, _ = predict_emotion(user_input, vectorizer, classifier)
            
            st.success(f"**Predicted Emotion:** {predicted_emotion.upper()}")
        else:
            st.warning("Please enter some text to analyze.")
else:
    st.error("Model assets failed to load. Please check the model file names and placement.")
