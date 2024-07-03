import streamlit as st
import pickle 
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(max_features=3000)





tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))


#transform text func
def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)

    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text=y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

st.sidebar.header(" About Model")
st.sidebar.success("""
                   SpamShield is an advanced machine learning-based solution designed 
                   to detect and filter out spam emails and SMS messages effectively. Leveraging  
                   robust machine learning algorithms, SpamShield offers a reliable defense 
                   against unwanted communications.""")
st.sidebar.header("key features")
st.sidebar.write("Text Preprocessing")
st.sidebar.write("Machine Learning Algorithms")
st.sidebar.write("Model Training and Evaluation")
st.sidebar.write("Real-Time Prediction")



st.title("SpamShield !!")
st.header("Email / SMS Spam Classifier 📧!!")
sms= st.text_area("Enter the text ..")
b1=st.button('Predict')

if b1:
   
    if sms:
        #preprocess
        transform_sms=transform_text(sms)

        #vectorize
        vector_input=tfidf.transform([transform_sms])

        #predict
        result=model.predict(vector_input[0])
        
    
        if result == 1:
            st.header("It is an spam Message 💢!!")
        else :
            st.header("It is an valid Message 💯 !!")
    else :
        st.header("Please enter the message !!")

