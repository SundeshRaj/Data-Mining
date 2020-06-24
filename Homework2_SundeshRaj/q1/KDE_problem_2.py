# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 22:48:39 2020

@author: sundesh raj
"""

import numpy as np
import matplotlib.pyplot as plt
from more_itertools import numeric_range
import matplotlib.patches as mpatches

def makeGaussianData2D (n, m, cov):
    np.random.seed(0)
    nData = np.random.multivariate_normal(m, cov, n)
    return nData

def mykde (X, h):
    prob = []
    n = len(X)
    Xi = tuple(numeric_range(min([min(r) for r in X]), max([max(s) for s in X]), .001))
    
    for x in Xi:
        sum_x = 0
        sum_y = 0
        for i in X:
            t1 = (x-i[0])/h
            t2 = (x-i[1])/h
            if abs(t1) <= 0.5:
                k = 1
            else:
                k = 0
            if abs(t2) <= 0.5:
                l = 1
            else:
                l = 0
            sum_x += k
            sum_y += l
        p1 = sum_x*(1/n*h)
        prob.append(p1)
        p2 = sum_y*(1/n*h)
        prob.append(p2)
    return prob

if __name__ == '__main__':
    
    #the means
    m1 = [1, 0]
    m2 = [0, 1.5]
    
    #the covariances
    coVariance1 = [[0.9,0.4],[0.4,0.9]]
    coVariance2 = [[0.9,0.4],[0.4,0.9]]
    
    X = makeGaussianData2D(500, m1, coVariance1)
#    X = np.array(list(zip(x, y)))
    
    Y = makeGaussianData2D(500, m2, coVariance2)
#    Y = np.array(list(zip(p, q)))
    
    randDataSet = np.concatenate((X, Y), axis=0)
    
    fig = plt.figure()
    fig2 = plt.figure()
    plt.rcParams["figure.figsize"] = (15,10)    
    
    # plotting with h=0.1
    ax_1 = fig.add_subplot(2, 2, 1)
    a = mykde(randDataSet, 0.1)
    ax_1.plot(a)
    
    # plotting with h=1
    ax_2 = fig.add_subplot(2, 2, 2)
    b = mykde(randDataSet, 1)
    ax_2.plot(b)
    
    # plotting with h=5
    ax_3 = fig.add_subplot(2, 2, 3)
    c = mykde(randDataSet, 5)
    ax_3.plot(c)
    
    # plotting with h=10
    ax_4 = fig.add_subplot(2, 2, 4)
    d = mykde(randDataSet, 10)
    ax_4.plot(d)
    
    # display gridlines 
    g1 = ax_1.grid(True)
    g2 = ax_2.grid(True)
    g3 = ax_3.grid(True)
    g4 = ax_4.grid(True)
    
    # Titles
    t1 = ax_1.set_title(r"Gaussian KDE")
    t2 = ax_2.set_title(r"Gaussian KDE")
    t3 = ax_3.set_title(r"Gaussian KDE")
    t4 = ax_4.set_title(r"Gaussian KDE")
    
    # LEGENDS
    l1 = mpatches.Patch(color=None, label='bandwidth=0.1')
    l2 = mpatches.Patch(color=None, label='bandwidth=1')
    l3 = mpatches.Patch(color=None, label='bandwidth=5')
    l4 = mpatches.Patch(color=None, label='bandwidth=10')
    
    ax_1.legend(handles=[l1])
    ax_2.legend(handles=[l2])
    ax_3.legend(handles=[l3])
    ax_4.legend(handles=[l4])
    
    # plotting the histogram for X -> data
    axH1 = fig2.add_subplot(2, 2, 1)
    axH1.hist(randDataSet)
    
    plt.tight_layout()
    plt.show()