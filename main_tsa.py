#!/usr/bin/env python
# coding: utf-8

# In[43]:


#import necessary libraries for data manipulation

import pandas as pd
import numpy as np
import re
import string
#import necessary libraries for 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings("ignore")


# In[5]:


#stopwords
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# In[17]:


#load the data
def load_data(filepath, cols):
    data = pd.read_csv(filepath, encoding='latin-1')
    data.columns = cols
    return data


# In[7]:


#remove redundant columns

def redundant_cols(data,cols):
    
    for col in cols:
        del data[col]
    return data


# In[11]:


#preprocess data

def preprocess_data(tweet):
    #convert text to lowercase
    tweet = tweet.lower()
    # remove any urls
    tweet = re.sub(f"http\S+|www\S+|https\S+", "", tweet, flags=re.MULTILINE)
    #remove puntuations
    tweet = tweet.translate(str.maketrans("","", string.punctuation))
    #remove user @ references and '#' from tweet
    tweet = re.sub(r'\@\w+|\#', "", tweet)
    #remove stopwords
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [word for word in tweet_tokens if word not in stop_words]
    #stemming
    ps = PorterStemmer()
    stemmed_words = [ps.stem(w) for w in filtered_words]
    #lemmatizing
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in stemmed_words]
    
    return " ".join(lemma_words)
    
preprocess_data("Hi there, how are you preparing for your exams?")


# In[12]:


#Vectorizing Tokens

def get_vector(train_fit):
    vector = TfidfVectorizer(sublinear_tf=True)
    vector.fit(train_fit)
    return vector


# In[18]:


# Load the dataset
filepath = 'twitter_data.csv'
cols = ['target', 'id', 'date', 'flag', 'user', 'text']
data = load_data(filepath, cols)


# In[19]:


data.head()


# In[20]:


# Remove redundant columns
redundant_columns = ['id', 'date', 'flag', 'user']
data = redundant_cols(data, redundant_columns)


# In[21]:


data.head()


# In[22]:


# Preprocess the tweets
data['text'] = data['text'].apply(preprocess_data)


# In[23]:


# Split the dataset
X = data['text']
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[44]:


# Train a Word2Vec model
word2vec_model = Word2Vec(sentences=X_train, vector_size=100, window=5, min_count=1, workers=4)
word2vec_model.train(X_train, total_examples=word2vec_model.corpus_count, epochs=10)


# In[45]:


# Convert text data to Word2Vec embeddings
X_train_vec = get_word2vec_embeddings(X_train, word2vec_model, 100)
X_test_vec = get_word2vec_embeddings(X_test, word2vec_model, 100)


# In[26]:


# Vectorize the text data
vector = get_vector(X_train)
X_train_vec = vector.transform(X_train)
X_test_vec = vector.transform(X_test)


# In[27]:


# Train the Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train_vec, y_train)


# In[28]:


# Train the Naive Bayes model
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(X_train_vec, y_train)


# In[29]:


# Make predictions with Logistic Regression
logistic_y_pred = logistic_model.predict(X_test_vec)


# In[30]:


# Make predictions with Naive Bayes
naive_bayes_y_pred = naive_bayes_model.predict(X_test_vec)


# In[33]:


# Evaluate the Logistic Regression model
logistic_accuracy = accuracy_score(y_test, logistic_y_pred)
logistic_report = classification_report(y_test, logistic_y_pred)


# In[34]:


# Evaluate the Naive Bayes model
naive_bayes_accuracy = accuracy_score(y_test, naive_bayes_y_pred)
naive_bayes_report = classification_report(y_test, naive_bayes_y_pred)


# In[35]:


print(f"Logistic Regression Accuracy: {logistic_accuracy}")
print("Logistic Regression Classification Report:")
print(logistic_report)


# In[36]:


print(f"Naive Bayes Accuracy: {naive_bayes_accuracy}")
print("Naive Bayes Classification Report:")
print(naive_bayes_report)


# In[37]:


import joblib


# In[39]:


# Save the models
joblib.dump(logistic_model, 'logistic_model.pkl')


# In[40]:


joblib.dump(naive_bayes_model, 'naive_bayes_model.pkl')


# In[41]:


joblib.dump(vector, 'tfidf_vectorizer.pkl')


# In[ ]:




