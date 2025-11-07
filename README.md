# Emotion-ray

[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-brightgreen)](https://emotion-ray-zo8mt666gjkxvcro86j8jb.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Active-blueviolet)

**A Streamlit app for classifying text into six emotions** (Anger, Fear, Joy, Love, Sadness, Surprise). Emotion-ray combines machine learning and NLP techniques to detect emotional tone in text, providing actionable insights for automation, personalization, or analytics.

---

## 🔹 Project Overview

Emotion-ray started as an experiment in text-based emotion detection and evolved into a robust classification pipeline using:

* **TF-IDF vectorization** for feature extraction
* **Support Vector Machine (SVM)** classifier
* **Data augmentation** (cross-lingual back-translation + rare-class adjustments)
* **Streamlit app** for interactive predictions

The project is designed to be **modular and deployable**, making it suitable for integration with email platforms, dashboards, or other automation tools.

---

## 📊 Model Performance

### Baseline: TF-IDF + SVM

| Emotion     | Precision | Recall | F1-Score | Support |
| ----------- | --------- | ------ | -------- | ------- |
| 😠 Anger    | 0.88      | 0.89   | 0.89     | 542     |
| 😨 Fear     | 0.89      | 0.87   | 0.88     | 475     |
| 😊 Joy      | 0.93      | 0.91   | 0.92     | 1352    |
| 💗 Love     | 0.74      | 0.87   | 0.80     | 328     |
| 😢 Sadness  | 0.94      | 0.92   | 0.93     | 1159    |
| 😲 Surprise | 0.76      | 0.81   | 0.79     | 144     |

**Overall:** Accuracy 0.90 | Macro F1 0.87 | Weighted F1 0.90

---

### Augmented: TF-IDF + SVM

| Emotion     | Precision | Recall | F1-Score | Support |
| ----------- | --------- | ------ | -------- | ------- |
| 😠 Anger    | 0.94      | 0.89   | 0.92     | 542     |
| 😊 Joy      | 0.95      | 0.95   | 0.95     | 475     |
| 😨 Fear     | 0.95      | 0.97   | 0.96     | 1352    |
| 😢 Sadness  | 0.92      | 0.93   | 0.92     | 328     |
| 😲 Surprise | 0.96      | 0.96   | 0.96     | 1159    |
| 💗 Love     | 0.94      | 0.92   | 0.93     | 144     |

**Overall:** Accuracy 0.95 | Macro F1 0.94 | Weighted F1 0.95

---

### 🏆 Baseline vs Augmented Comparison

| Metric                         | Baseline    | Augmented   | Improvement   |
| ------------------------------ | ----------- | ----------- | ------------- |
| Accuracy                       | 0.90        | 0.95        | +0.05         |
| Macro Avg F1                   | 0.87        | 0.94        | +0.07         |
| Weighted Avg F1                | 0.90        | 0.95        | +0.05         |
| Rare-class (Love, Surprise) F1 | 0.80 / 0.79 | 0.93 / 0.96 | +0.13 / +0.17 |

**Highlights:**

* Augmentation improved recognition for underrepresented emotions
* Overall, the augmented model is more balanced and reliable

---

## ⭐ Visual Summary: F1 Score Improvements (Star-style)

### Per Emotion

| Emotion     | Baseline F1 | Augmented F1 | Visual        |
| ----------- | ----------- | ------------ | ------------- |
| 😠 Anger    | 0.89        | 0.92         | ★★★★☆ → ★★★★★ |
| 😨 Fear     | 0.88        | 0.96         | ★★★★☆ → ★★★★★ |
| 😊 Joy      | 0.92        | 0.95         | ★★★★★ → ★★★★★ |
| 💗 Love     | 0.80        | 0.93         | ★★★★☆ → ★★★★★ |
| 😢 Sadness  | 0.93        | 0.92         | ★★★★★ → ★★★★★ |
| 😲 Surprise | 0.79        | 0.96         | ★★★★☆ → ★★★★★ |

### Overall Metrics

| Metric      | Baseline | Augmented | Visual        |
| ----------- | -------- | --------- | ------------- |
| Accuracy    | 0.90     | 0.95      | ★★★★★ → ★★★★★ |
| Macro F1    | 0.87     | 0.94      | ★★★★☆ → ★★★★★ |
| Weighted F1 | 0.90     | 0.95      | ★★★★★ → ★★★★★ |

**Legend:** ★ = ~0.20 F1; arrows show baseline → augmented

---

## 🧩 Usage

**Load models locally:**

```python
import joblib

vec = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("svm_emotion_classifier.pkl")

sample_text = "I’m so excited about our new project!"
X = vec.transform([sample_text])
prediction = model.predict(X)[0]
print(prediction)
```

**Live Demo:** [Try the Emotion-ray Streamlit App](https://emotion-ray-zo8mt666gjkxvcro86j8jb.streamlit.app/)

---

## 🌿 Future Plans

* **Model Improvements:** Refine detection using better data balance, context-aware embeddings (e.g., DistilBERT), and multilingual support
* **Email Automation Integration:** Detect user mood from messages → personalize email tone or timing
* **Analytics Dashboard:** Track emotional trends over time for actionable insights

---

## 🗂 Project Structure

```
Emotion-ray/
├── app.py
├── requirements.txt
├── tfidf_vectorizer.pkl
├── svm_emotion_classifier.pkl
└── README.md
```

---

This version now:

* **Highlights incremental improvements** clearly with stars
* Keeps all tables, visuals, and performance metrics intact
* Includes badges, live demo, usage instructions, and future plans

---

If you want, I can also **suggest one small tweak to make it even more portfolio-friendly**: adding **mini emojis or colored labels to the tables** for immediate visual cues (like green ↑ for improvement, red ↓ for drop).

Do you want me to do that?

---

### ⭐ Visual Summary: F1 Score Improvements (Star + Arrows)

| Emotion     | Baseline F1 | Augmented F1 | Visual            |
| ----------- | ----------- | ------------ | ----------------- |
| 😠 Anger    | 0.89        | 0.92         | ★★★★☆ → ★★★★★ 🟢↑ |
| 😨 Fear     | 0.88        | 0.96         | ★★★★☆ → ★★★★★ 🟢↑ |
| 😊 Joy      | 0.92        | 0.95         | ★★★★★ → ★★★★★ 🟢↑ |
| 💗 Love     | 0.80        | 0.93         | ★★★★☆ → ★★★★★ 🟢↑ |
| 😢 Sadness  | 0.93        | 0.92         | ★★★★★ → ★★★★★ 🔴↓ |
| 😲 Surprise | 0.79        | 0.96         | ★★★★☆ → ★★★★★ 🟢↑ |

---

### Overall Metrics (Star + Arrows)

| Metric      | Baseline | Augmented | Visual            |
| ----------- | -------- | --------- | ----------------- |
| Accuracy    | 0.90     | 0.95      | ★★★★★ → ★★★★★ 🟢↑ |
| Macro F1    | 0.87     | 0.94      | ★★★★☆ → ★★★★★ 🟢↑ |
| Weighted F1 | 0.90     | 0.95      | ★★★★★ → ★★★★★ 🟢↑ |

**Legend:**

* ★ = ~0.20 F1
* Arrows + color indicate change (🟢↑ = improvement, 🔴↓ = slight decrease)
* Arrows show baseline → augmented

---


