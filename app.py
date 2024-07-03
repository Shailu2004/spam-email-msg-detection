import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources (only needs to be done once)
nltk.download('punkt')
nltk.download('stopwords')

# Load pickled models
with open('vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize PorterStemmer
ps = PorterStemmer()

# Transform text function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Sidebar content
st.sidebar.header("About Model")
st.sidebar.success("""
                   SpamShield is an advanced machine learning-based solution designed 
                   to detect and filter out spam emails and SMS messages effectively. Leveraging  
                   robust machine learning algorithms, SpamShield offers a reliable defense 
                   against unwanted communications.""")
st.sidebar.header("Key Features")
st.sidebar.write("Text Preprocessing")
st.sidebar.write("Machine Learning Algorithms")
st.sidebar.write("Model Training and Evaluation")
st.sidebar.write("Real-Time Prediction")

# Main content
st.title("SpamShield !!")
st.header("Email / SMS Spam Classifier 📧!!")
sms = st.text_area("Enter the text ..")
b1 = st.button('Predict')

if b1:
    if sms:
        # Preprocess
        transform_sms = transform_text(sms)

        # Vectorize
        vector_input = tfidf.transform([transform_sms])

        # Predict
        result = model.predict(vector_input)[0] 
        if result == 1:
            st.header("It is a spam Message 💢!!")
        else:
            st.header("It is a valid Message 💯 !!")
    else:
        st.header("Please enter the message !!")
