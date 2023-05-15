import streamlit as st
import numpy as np
import pandas as pd
import sklearn
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

st.set_page_config(
    page_title="Spam SMS Classifier",
    page_icon="x",
    layout="wide"
)

nltk.download("stopwords")
stwords = stopwords.words('english')
df = pd.read_csv("data.csv")

def clean_data(sentence):
    """Cleaning the datas"""
    # Removing the symbols and digits
    word_list = np.array(re.sub(r'[^a-zA-Z]', ' ', sentence).split())

    # Lowering the text data
    word_list = np.array([word.lower() for word in word_list])

    # Removing the stopwords
    word_list = np.array([word for word in word_list if word not in stwords])
    
    # Stemming the words
    stemmer = PorterStemmer()
    word_list = np.array([stemmer.stem(word) for word in word_list])

    # Joinging all word to make sentence
    return " ".join(word_list)


model = pickle.load(open("model.pkl", "rb"))
vec = pickle.load(open("vect.pkl", 'rb'))

def predict(sentence):
    sentence = np.array([sentence])
    sentence = vec.transform(sentence)
    result = model.predict(sentence)
    return result



st.title("Spam SMS Classifier")
text = st.text_input("Enter a SMS", \
                     placeholder="eg. Free entry in 2 a wkly comp to win FA Cup",\
                        )
if text != "":
    res = predict(text.strip())
    col1, col2 = st.columns([1, 10])
    col1.subheader("Result: ")
    if res == 'spam':
        col2.error("Spam 🚨")
    else:
        col2.success("Ham 🔥")

st.write("Example SMS of  :red[SPAM]")
st.code(f"""{df[df['label'] == 'spam']['text'].sample(1).values[0]}""")
st.caption("Example SMS of  :green[HAM]")
st.code(f"""Ham: {df[df['label'] == 'ham']['text'].sample(1).values[0]}""")


