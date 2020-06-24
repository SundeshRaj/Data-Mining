# -*- coding: utf-8 -*-
"""
Created on Wed Feb  26 18:54:43 2020

@author: sundesh raj
"""

import numpy as np
import copy as cp
import matplotlib.pyplot as plt

#p1-1


#this kmeans function is referenced from https://github.com/sagar3122/Data-Mining/blob/master/K-Means/k-means.py
def myKmeans(X,k,c):
    
    dataLenth = X.shape[0]
    old_center = np.zeros(c.shape)
    new_center = cp.deepcopy(c)
    clusters = np.zeros(dataLenth)
    distance = np.zeros((dataLenth,k))
    stop = np.linalg.norm(new_center-old_center)
    i = 0
    while stop > 0.001:
        i = i+1
        if i == 10000:
            break
        for x in range(k):
            distance[:, x] = np.linalg.norm(X - new_center[x], axis = 1)
        clusters = np.argmin(distance, axis = 1)
        old_center = cp.deepcopy(new_center)
        for x in range(k):
            new_center[x] = np.mean(X[clusters == x], axis = 0 )
        stop = np.linalg.norm(new_center - old_center)
    print("Number of iterations when k={} is {}".format(k,i))
    print("Centers for the clusters after performing kMeans and when k={} is \n{}".format(k,new_center))

    #scatter plot
    xplots = []
    yplots = []
    
    for val in X:
        for data in val:
            if data == val[0]:
                xplots.append(data)
            if data == val[1]:
                yplots.append(data)
                
    for plot,x,y in zip(clusters, xplots, yplots):
        if plot == 0:
            plt.scatter(x, y, color = 'grey')
            continue
        elif plot == 1:
            plt.scatter(x, y, color = 'cyan')
            continue
        elif plot == 2:
            plt.scatter(x, y, color = 'red')
            continue
        elif plot == 3:
            plt.scatter(x, y, color = 'blue')
            continue

    for x,y in new_center:
        plt.scatter(x, y, color = 'black')

    plt.xlabel("xplots")
    plt.ylabel("yplots")
    plt.show()


if __name__ == '__main__':
   
    #the means
    m1 = [1, 0]
    m2 = [0, 1.5]
    
    #the covariances
    coVariance1 = [[0.9,0.4],[0.4,0.9]]
    coVariance2 = [[0.9,0.4],[0.4,0.9]]
    
    np.random.seed(0)
    
    randData1 = np.random.multivariate_normal(m1, coVariance1, 500)
    randData2 = np.random.multivariate_normal(m2, coVariance2, 500)
    
    randDataSet = np.concatenate((randData1, randData2))
    
    print("##################################################################")
    print("      Problem 1 part 2, k=2 Centers = [10,10] and [-10,-10]       ")
    print("##################################################################")
    
    k1 = 2;
    clusterCenters1 = [[10, 10],
                      [-10, -10]]
    clusterCenters1 = np.asarray(clusterCenters1, dtype = np.float64)
    
    myKmeans(randDataSet, k1, clusterCenters1)
    
    print("################################################################################")
    print("      Problem 1 part 3, k=4 Centers = [10,10],[-10,-10],[10,-10],[-10,10]       ")
    print("################################################################################")
          
    k2 = 4
    clusterCenters2 = [[10, 10],
                      [-10, -10],
                      [10, -10],
                      [-10, 10]]
    clusterCenters2 = np.asarray(clusterCenters2, dtype = np.float64)
    
    myKmeans(randDataSet, k2, clusterCenters2)
    