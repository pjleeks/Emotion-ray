import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords

# --- FORCE PUNKT DOWNLOAD ---
# This line is added to guarantee the 'punkt' tokenizer is available for word_tokenize.
# We are forcing this download as conditional checks have been unreliable.
nltk.download('punkt', quiet=True) 
# ----------------------------

# --- 1. CONFIGURATION AND ASSET LOADING ---

# Define the file names (These must be in the same directory as this file)
MODEL_FILENAME = 'svm_emotion_classifier.pkl'
VECTORIZER_FILENAME = 'tfidf_vectorizer.pkl'

# --- Define the constants (Emotion Map and Preprocessing tools) ---
emotion_map = {
    0: 'sadness', 1: 'joy', 2: 'love', 
    3: 'anger', 4: 'fear', 5: 'surprise'
}

# Download stopwords if not present (This conditional is necessary for Streamlit Cloud)
try:
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
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
    """
    Applies the exact same preprocessing as done during training, 
    using nltk.word_tokenize which requires the downloaded 'punkt' data.
    """
    if not isinstance(text, str):
        return "" 
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Tokenize using NLTK's word_tokenize
    words = nltk.word_tokenize(text)
    
    # 3. Remove Punctuation and Stopwords
    words = [
        word for word in words 
        if word not in PUNCTUATION and word not in STOPWORDS
    ]
    
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
