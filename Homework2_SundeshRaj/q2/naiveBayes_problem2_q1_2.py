# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 23:25:21 2020

@author: sundesh raj
"""

from naiveBayes import NaiveBayes



if __name__ == '__main__':
    
    # means
    m1 = [1,0]
    m2 = [0,1]
    
    # std deviations
    stdDev1 = [[1, 0.75],[0.75, 1]]
    stdDev2 = [[1, 0.75],[0.75, 1]]

    # data variables for problem
    set1 = 1
    randItemIterator1 = 2

    set2 = 6
    randItemIterator2 = 3
    
    set3 = 1
    randItemIterator3 = 4
    
    print()
    print("############################################################################")
    print("-------------------------Problem 2 question 1a/2----------------------------")
    print("---Testing the prediction on the test data along with scatter plot and ROC--")
    print("############################################################################")
    NaiveBayes(set1, randItemIterator1, 1, 1, m1, m2, stdDev1, stdDev2)
    
#    print()
#    print("############################################################################")
#    print("-------------------------Problem 2 question 1b------------------------------")
#    print("---------Changing number of samples [10, 20, 50, 100, 300, 500]-------------")
#    print("############################################################################")
#    NaiveBayes(set2, randItemIterator2, 0, 0, m1, m2, stdDev1, stdDev2)
#    
    print()
    print("############################################################################")
    print("-------------------------Problem 2 question 1c/2----------------------------")
    print("-----With 300 items for label 1 and 700 for label 0 with ROC and scatter----")
    print("############################################################################")
    NaiveBayes(set3, randItemIterator3, 1, 1, m1, m2, stdDev1, stdDev2)
    