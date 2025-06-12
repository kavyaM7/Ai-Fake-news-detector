import os
import streamlit as st
import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Title and UI
st.set_page_config(page_title="AI Fake News Detector", layout="centered")
st.title("📰 AI Fake News Detector")
st.markdown("Enter a news article below to check whether it's real or fake.")

# Load and prepare dataset
@st.cache_data
def load_data():
    df_fake = pd.read_csv("Fake.csv")
    df_true = pd.read_csv("True.csv")
    
    df_fake["label"] = 0  # 0 for fake
    df_true["label"] = 1  # 1 for real
    
    data = pd.concat([df_fake, df_true], axis=0)
    data = data[["text", "label"]]
    data.dropna(inplace=True)
    return data

data = load_data()

# Vectorizer and Model setup
@st.cache_resource
def train_model(data):
    tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
    X = tfidf.fit_transform(data['text'])
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = PassiveAggressiveClassifier(max_iter=1000)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, tfidf, acc

model, tfidf, acc = train_model(data)

# Input
user_input = st.text_area("Enter News Text Here:", height=150)

if st.button("Detect"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        input_data = tfidf.transform([user_input])
        prediction = model.predict(input_data)[0]
        
        if prediction == 0:
            st.error("🔴 This news is *FAKE*.")
        else:
            st.success("🟢 This news is *REAL*.")
        
        st.info(f"Model Accuracy: *{acc * 100:.2f}%*")