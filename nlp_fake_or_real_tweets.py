# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 14:23:47 2020

@author: Lenovo
"""

#NLP for Decising the Tweets are Real or Fake?

#Importing the Libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the Datasets:
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

#Feature Engineering and Cleaning:

#Dropping the keyword and location column vector:
df_train = df_train.drop(labels = ['keyword', 'location'], axis = 1) 
#print(df_train.info())
df_test = df_test.drop(labels = ['keyword','location'], axis = 1)


#Importing the ImP Libraries:
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
ps = PorterStemmer()
"""
#Cleaning of 1st Row Vector:
review = re.sub('[^a-zA-Z]', ' ', df_train['text'][0])
review = review.lower()
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
review = ' '.join(review)
"""
"""
#Cleaning of Entire training Datasets:
corpus_1 = [ ]
for i in range(0, 7613):
    review_1 = re.sub('[^a-zA-Z]',' ',df_train['text'][i]) 
    review_1 = review_1.lower()
    review_1 = review_1.split()
    review_1 = [ps.stem(word) for word in review_1 if not word in set(stopwords.words('english'))]
    review_1 = ' '.join(review_1)
    corpus_1.append(review_1)

#Cleaning of Entire Testing Datasets:
corpus_2 = [ ]
for i in range(0, 3263):
    review_2 = re.sub('[^a-zA-Z]',' ', df_test['text'][i])
    review_2 = review_2.lower()
    review_2 = review_2.split()
    review_2 = [ps.stem(word) for word in review_2 if not word in set(stopwords.words('english'))]
    review_2 = ' '.join(review_2)
    corpus_2.append(review_2)
       

#Creating the Bag of Words Model:
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 10766)

#Fitting on Training Set:
x = cv.fit_transform(corpus_1).toarray()
y = df_train.iloc[:, -1].values

#Fitting on Testing Set:
x_test = cv.fit_transform(corpus_2).toarray()


"""
#Building the Classification Model Hereafter:
"""
#Training of SVC Classifier:
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 500,criterion ="entropy" )
classifier.fit(x,y)
"""
#Training the Naive Bayes Model:
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x,y)

#Predicting the Results:
y_pred = classifier.predict(x_test)
