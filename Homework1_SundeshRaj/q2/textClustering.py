# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 10:15:44 2020

@author: sundesh raj
"""

import pandas as pd
import re
from kMeans import myKmeans
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#import matplotlib.pyplot as plt
#import operator
#import math

if __name__ == '__main__':
    
    reviewsDataFile = "Amazon_Reviews.csv"
    reviewsDataSet = pd.read_csv(reviewsDataFile)
    
    corpusColumn = []
    
    for i in range(reviewsDataSet.shape[0]):
        
        amzReviews = re.sub('[^a-zA-Z]', ' ', reviewsDataSet['Review'][i]) # cleaning rows using re
        amzReviews = amzReviews.lower() # convert to lowercase
        amzReviews = amzReviews.split() # split at " "
        
        pStemmer = PorterStemmer()
        
        #looping for stemming each word
        amzReviews = [pStemmer.stem(word) for word in amzReviews if not word in set(stopwords.words('english'))]
        
        amzReviews = ' '.join(amzReviews)
        
        corpusColumn.append(amzReviews)
        
    
    tfidfVectorizer = TfidfVectorizer()
    tfIdf = tfidfVectorizer.fit_transform(corpusColumn)
    features = tfidfVectorizer.get_feature_names()
    makeDense = tfIdf.todense()
    tfIdfDense = makeDense.tolist()
    
    print("##################################################################")
    print("       Problem 2 part 1, Visualize the tf-idf weight matrix       ")
    print("##################################################################")
          
    ind = pd.RangeIndex(199)
    dataFrame = pd.DataFrame(tfIdfDense, index=ind, columns=features)
    print("\n\n The shape of the tf-idf matrix after stemming and removing stop words:\n{}".format(dataFrame.shape))
    print("\nVisualizing the tf-idf matrix in 2D with color code \n")
    sns.heatmap(dataFrame, cmap='CMRmap_r')
    
    print()
    print()
    print("##################################################################")
    print(" Problem 2 part 2, 5 positive and negative words count and tf-idf ")
    print("##################################################################")
          
          
    #5 positvie words
    # brilliant, awesom, good, nice, perfect
    
    #5 negative words
    # absurd, annoy, bad, disappoint, poor
    
    cntVectorizer = CountVectorizer()
    cnt = cntVectorizer.fit_transform(corpusColumn)
    cntFeatures = cntVectorizer.get_feature_names()
    cntDense = cnt.todense()
    countMatrix = cntDense.tolist()
    countDF = pd.DataFrame(countMatrix, index=ind, columns=cntFeatures)
    
    #get the matrix of only the positive and negative words as features
    posNegCountMatrix = countDF[['brilliant', 'awesom', 'good', 'nice', 'perfect', 'absurd', 'annoy', 'bad', 'disappoint', 'poor']]
    print()
    print()
    print("Listing the positive and negative words selected:\n")
    print(posNegCountMatrix.columns)
    
    print()
    print()
    print("Printing the first 10 elements from the count matrix\n")
    # to print all the 199 rows remove the .head(10) method
    print(posNegCountMatrix.head(10))
    
    print()
    print("Printing the first 10 tf-idf matrix elements for the selected positive and negative words\n")
    posNegtfIdfDF = dataFrame[['brilliant', 'awesom', 'good', 'nice', 'perfect', 'absurd', 'annoy', 'bad', 'disappoint', 'poor']]
    # to print all the 199 rows remove the .head(10) method
    print(posNegtfIdfDF.head(10))
    
    print()
    print("Visualizing the tf-idf matrix for the positive and negative words")
    sns.heatmap(posNegtfIdfDF, cmap='CMRmap_r')
    
    print()
    print()
    print("##################################################################")
    print(" Problem 2 part 3, 5 positive and negative words count and tf-idf ")
    print("##################################################################")
    
          
    print()
    newCluster = []
    p = 0
    n = 0
    for i in range(posNegCountMatrix.shape[0]):
        p = posNegCountMatrix.loc[i][:5].sum(axis=0)
        n = posNegCountMatrix.loc[i][5:].sum(axis=0)
        newCluster.append([p,n])
        
    print()
    newClusterDF = pd.DataFrame(newCluster)
    newClusterDF = np.asarray(newClusterDF, dtype=np.float64)
    
    #k means for k=2/3/4
    for k in [2,3,4]:
        if k==2:
            c1 = [[0, 2],
                  [2, 0]]
            c1 = np.asarray(c1, dtype = np.float64)
            myKmeans(newClusterDF, 3, c1)
        
        if k==3:
            c2 = [[0, 1],
                  [1, 0],
                  [0, -1]]
            c2 = np.asarray(c2, dtype = np.float64)
            myKmeans(newClusterDF, k, c2)
            
        if k==4:
            c3 = [[0, 1],
                  [1, 0],
                  [0, -1],
                  [-1, 0]]
            c3 = np.asarray(c3, dtype = np.float64)
            myKmeans(newClusterDF, k, c3)
        
    
    
