import pandas as pd
import numpy as np
import seaborn as sns
import nltk as nltk
from nltk.tokenize import word_tokenize
import pdfplumber as pdf
import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer

df=pd.read_csv(r"C:\Users\Legend\Downloads\UpdatedResumeDataSet.csv")


print(df.columns.tolist())
print(df.head(3))

#cleaning the data set

def clean_text(text):
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = text.lower()
    return text

df['Resume'] = df['Resume'].apply(clean_text)

print(df.head())


# train the model
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle


X = df["Resume"]
y = df["Category"]

tfidf = TfidfVectorizer(stop_words='english')

X_vec = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(tfidf, open("vectorizer.pkl", "wb"))


# test prediction
import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

resume = """
Python, Machine Learning, SQL,
Data Analysis, Pandas, NumPy
"""

data = vectorizer.transform([resume])

prediction = model.predict(data)

print("Predicted Category:", prediction[0])


#built the web page

import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("AI Resume Category Predictor")

resume_text = st.text_area(
    "Paste Resume Text Here",
    height=300
)

if st.button("Predict Category"):

    if resume_text.strip():

        vector = vectorizer.transform([resume_text])

        prediction = model.predict(vector)

        probabilities = model.predict_proba(vector)

        st.success(
            f"Predicted Category: {prediction[0]}"
        )
        confidence = max(probabilities[0]) * 100

        st.write(f"Confidence: {confidence:.2f}%")

    else:
        st.warning("Please enter resume text.")



# uplode resume pdf
uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)