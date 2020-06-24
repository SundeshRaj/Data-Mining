# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 22:42:18 2020

@author: sundesh raj
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

if __name__ == '__main__':
    
    reviewsDataFile = "Amazon_Reviews.csv"
    reviewsDataSet = pd.read_csv(reviewsDataFile)
    reviewsDataSet['split'] = np.random.randn(reviewsDataSet.shape[0], 1)
    
    filterM = np.random.rand(len(reviewsDataSet)) <= 0.5

    trainData = reviewsDataSet[filterM]
    testData = reviewsDataSet[filterM]

    X_Train = trainData['Review']
    Y_Train = trainData['Label']
    X_Test = testData['Review']
    Y_Test = testData['Label']
    
    X = np.concatenate((X_Train, X_Test), axis = 0)
    Y = np.concatenate((Y_Train, Y_Test), axis = 0)
    
    # TF-IDF Vecotorizer
    tfIdfVectorizer = TfidfVectorizer()
    
    X_TF = tfIdfVectorizer.fit_transform(X)
    
    NBClassifier = MultinomialNB()
    
    kFold = KFold(5)
    
    count = 1
    for trainIndex, testIndex in kFold.split(X_TF):
        
        X_train, X_test = X_TF[trainIndex], X_TF[testIndex]
        Y_train, Y_test = Y[trainIndex], Y[testIndex]
        
        NBClassifier.fit(X_train, Y_train)
        
        predictedValues = NBClassifier.predict(X_test)
        
        accuracyScore = metrics.accuracy_score(Y_test, predictedValues)
        
        print("############################################################################")
        print("---------------------Accuracy score for fold {} is {}-----------------------".format(count, accuracyScore))
        print("############################################################################")
        count += 1
        print()
        print("############################################################################")
        print("--------------------------Classification Report-----------------------------")
        print(metrics.classification_report(Y_test, predictedValues, target_names = ['Positive', 'Negative']))
        print("############################################################################")
        
        print()
        print("############################################################################")
        print("------------------------------Confustion Matrix-----------------------------")
        print(metrics.confusion_matrix(Y_test, predictedValues))
        print("############################################################################")
