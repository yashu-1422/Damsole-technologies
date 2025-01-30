import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

# Load pre-trained models and vectorizer
models = {
    "Naive Bayes": joblib.load("Naive Bayes.pkl"),
    "Random Forest": joblib.load("Random Forest.pkl"),
    "KNN": joblib.load("KNN.pkl"),
    "XGBoost": joblib.load("XGBoost.pkl"),
    "Logistic Regression": joblib.load("Logistic Regression.pkl"),
    "Decision Tree": joblib.load("Decision Tree.pkl"),
}

vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Data Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r'\d+', '', text)
    return text

# Streamlit Interface
st.title('Sentiment Analysis of Movie Reviews')

st.write("""
This application predicts the sentiment of a movie review (positive/negative) using various machine learning models.
You can enter a movie review below, and the system will predict whether it's positive or negative.
""")

# Input from the user
user_input = st.text_area("Enter your movie review:")

# Function to make prediction
def predict_sentiment(review):
    review = preprocess_text(review)
    review_tfidf = vectorizer.transform([review])
    predictions = {}
    
    for model_name, model in models.items():
        prediction = model.predict(review_tfidf)
        predictions[model_name] = "Positive" if prediction == 1 else "Negative"
    
    return predictions

if st.button('Predict Sentiment'):
    if user_input:
        predictions = predict_sentiment(user_input)
        st.write("Sentiment Predictions:")
        for model_name, sentiment in predictions.items():
            st.write(f"{model_name}: {sentiment}")
    else:
        st.warning("Please enter a movie review to get the prediction.")
