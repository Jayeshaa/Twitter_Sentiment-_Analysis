import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Load the models and vectorizer
logistic_model = joblib.load('logistic_model.pkl')
naive_bayes_model = joblib.load('naive_bayes_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocess the input text (same as your preprocessing function)
def preprocess_data(tweet):
    import re
    import string
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    
    # Define stopwords
    stop_words = set(stopwords.words('english'))
    
    # Convert text to lowercase
    tweet = tweet.lower()
    # Remove any URLs
    tweet = re.sub(f"http\S+|www\S+|https\S+", "", tweet, flags=re.MULTILINE)
    # Remove punctuations
    tweet = tweet.translate(str.maketrans("", "", string.punctuation))
    # Remove user @ references and '#' from tweet
    tweet = re.sub(r'\@\w+|\#', "", tweet)
    # Remove stopwords
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [word for word in tweet_tokens if word not in stop_words]
    # Stemming
    ps = PorterStemmer()
    stemmed_words = [ps.stem(w) for w in filtered_words]
    # Lemmatizing
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in stemmed_words]
    
    return " ".join(lemma_words)

# Streamlit app
st.title('Text Classification App')

input_text = st.text_area("Enter the text to classify:")

if st.button("Classify"):
    if input_text:
        # Preprocess and vectorize the input text
        processed_text = preprocess_data(input_text)
        vectorized_text = vectorizer.transform([processed_text])
        
        # Predict with both models
        logistic_pred = logistic_model.predict(vectorized_text)
        naive_bayes_pred = naive_bayes_model.predict(vectorized_text)
        
        # Display predictions
        st.write(f"Logistic Regression Prediction: {'Positive' if logistic_pred[0] == 1 else 'Negative'}")
        st.write(f"Naive Bayes Prediction: {'Positive' if naive_bayes_pred[0] == 1 else 'Negative'}")
    else:
        st.write("Please enter some text to classify.")

