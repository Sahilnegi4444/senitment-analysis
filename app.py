import streamlit as st
import numpy as np
import pickle
from joblib import load

# ----------------------
# Load preprocessing + model
# ----------------------

from nltk.stem import WordNetLemmatizer
def lemmatize_words(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join(lemmatizer.lemmatize(word) for word in text.split())

with open('dim_reduction.pkl', 'rb') as dim_reduce:
    dim_reduction = pickle.load(dim_reduce)

with open('tf_idf.pkl', 'rb') as tokenize:
    tokenize_words = pickle.load(tokenize)

model = load("model.joblib")

# ----------------------
# Streamlit App UI
# ----------------------

st.set_page_config(page_title='Sentiment Analysis App', layout='centered')

st.title("Sentiment Analysis")
st.write("Enter a review to find out whether its **positive** or **Negative**")

#User Input
user_input = st.text_area("Enter your review here:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == '':
        st.warning("Please enter some text before predicting.")
    
    else:
        try:
            #Preprocess input(lemmatize) -> tf-idf -> dim reduction
            lem = lemmatize_words(user_input)
            X_tfidf = tokenize_words.transform([lem])
            X_svd = dim_reduction.transform(X_tfidf)

            #Predict
            pred = model.predict(X_svd)[0]
            prob = model.predict_proba(X_svd)[0]

            #Dispaly
            sentiment = "Positive" if pred == 1 else "Negative"
            st.subheader(f"Sentiment: {sentiment}")
            st.write(f"Prediction probability: {prob[pred]*100:.2f}%")

            #Display both negative and positive probab 
            st.write({
                "Negative":f"{prob[0]*100:.2f}%",
                "Positive":f"{prob[1]*100:.2f}%"
            })
        except Exception as e:
            st.error(f"Error during prediction{e}")
