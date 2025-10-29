import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords

# --- 1. CONFIGURATION AND ASSET LOADING ---

# Define the file names (These must be in the same directory as this file)
MODEL_FILENAME = 'svm_emotion_classifier.pkl'
VECTORIZER_FILENAME = 'tfidf_vectorizer.pkl'

# --- Define the constants (Emotion Map and Preprocessing tools) ---
emotion_map = {
    0: 'sadness', 1: 'joy', 2: 'love', 
    3: 'anger', 4: 'fear', 5: 'surprise'
}

# Download stopwords and punkt if not present (Crucial for deployment environments)
try:
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    # --- CRITICAL FIX: DOWNLOAD BOTH REQUIRED NLTK PACKAGES ---
    nltk.download('stopwords')
    nltk.download('punkt')  # <--- NEW LINE ADDED TO FIX TOKENIZATION
    STOPWORDS = set(stopwords.words('english'))
PUNCTUATION = string.punctuation


# --- Caching the Load Process for Speed ---
@st.cache_resource
def load_assets():
    """Loads the pickled model and vectorizer only once."""
    try:
        with open(MODEL_FILENAME, 'rb') as file:
            loaded_model = pickle.load(file)
        with open(VECTORIZER_FILENAME, 'rb') as file:
            loaded_vectorizer = pickle.load(file)
        return loaded_model, loaded_vectorizer
    except FileNotFoundError:
        st.error("Error: Model or Vectorizer files not found. Ensure .pkl files are in the directory.")
        return None, None

loaded_model, loaded_vectorizer = load_assets()


# --- 2. PREDICTION FUNCTIONS ---

def clean_text_for_prediction(text):
    """Applies the exact same preprocessing as done during training."""
    if not isinstance(text, str):
        return "" # Handle non-string input
    text = text.lower()
    text = "".join([char for char in text if char not in PUNCTUATION])
    words = text.split()
    words = [word for word in words if word not in STOPWORDS]
    return " ".join(words)

def predict_emotion(raw_text, model, vectorizer):
    """Takes raw text and returns the predicted emotion string."""
    if not model or not vectorizer:
        return "Model not loaded."
        
    cleaned_text = clean_text_for_prediction(raw_text)
    text_vec = vectorizer.transform([cleaned_text])
    prediction_label = model.predict(text_vec)[0]
    
    return emotion_map.get(prediction_label, "Unknown Emotion")


# --- 3. STREAMLIT FRONT-END ---

st.title("🧠 89% Accurate Text Emotion Classifier")
st.markdown("Enter any short text to predict one of six emotions: Sadness, Joy, Love, Anger, Fear, or Surprise.")

# Text input box
user_input = st.text_area("Type your text here:", "I can't believe I got an 89% accuracy on my first project!")

if st.button("Analyze Emotion"):
    if user_input:
        with st.spinner('Analyzing...'):
            # Predict the emotion using the loaded assets
            emotion = predict_emotion(user_input, loaded_model, loaded_vectorizer)
            
            # Display result with styling
            st.success(f"**Predicted Emotion:** {emotion.upper()}")
            
            # Optional: Display a visual based on emotion
            if emotion in ['joy', 'love']:
                st.balloons()
            elif emotion in ['anger', 'fear', 'sadness']:
                st.warning("Take a deep breath. We detected a negative emotion.")
    else:
        st.error("Please enter some text to analyze.")
