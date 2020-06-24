# -*- coding: utf-8 -*-
"""
Created on Tue Mar  24 11:47:14 2020

@author: sundesh raj
"""

import numpy as np
import matplotlib.pyplot as plt
from more_itertools import numeric_range
import matplotlib.patches as mpatches

def makeGaussianData1D (n, mean, stdDev):
    np.random.seed(0)
    mData = np.random.normal(mean, stdDev, n)
    return mData

def mykde (X, h):
    prob = []
    n = len(X)
    Xi = tuple(numeric_range(min(X), max(X), .001))
    
    for x in Xi:
        sum_x = 0
        for i in X:
            t = (x-i)/h
            if abs(t) <= 0.5:
                k = 1
            else:
                k = 0
            sum_x += k
        p = sum_x*(1/n*h)
        prob.append(p)
    return prob

if __name__ == '__main__':
    
    gRandData1a = makeGaussianData1D(1000, 5, 1) # 1a data
    fig = plt.figure()
    fig2 = plt.figure()
    plt.rcParams["figure.figsize"] = (15,10)    
    
    # plotting with h=0.1
    ax_1 = fig.add_subplot(2, 2, 1)
    a = mykde(gRandData1a, 0.1)
    ax_1.plot(a)
    
    # plotting with h=1
    ax_2 = fig.add_subplot(2, 2, 2)
    b = mykde(gRandData1a, 1)
    ax_2.plot(b)
    
    # plotting with h=5
    ax_3 = fig.add_subplot(2, 2, 3)
    c = mykde(gRandData1a, 5)
    ax_3.plot(c)
    
    # plotting with h=10
    ax_4 = fig.add_subplot(2, 2, 4)
    d = mykde(gRandData1a, 10)
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
    axH1.hist(gRandData1a)
    
    plt.tight_layout()
    plt.show()
    
    # end 1a
    
    # 1b start
    
    fig3 = plt.figure()
    fig4 = plt.figure()
    
    gRandData1b1 = makeGaussianData1D(1000, 5, 1) # 1b first set of Gaussian data
    
    gRandData1b2 = makeGaussianData1D(1000, 0, 0.2) # 1b second set of Gaussian data
    
    gRandData1b = np.concatenate((gRandData1b1, gRandData1b2), axis=0) # concat both the data
    
    # plotting with h=0.1
    ax_5 = fig3.add_subplot(2, 2, 1)
    e = mykde(gRandData1b, 0.1)
    ax_5.plot(e)
    
    # plotting with h=1
    ax_6 = fig3.add_subplot(2, 2, 2)
    f = mykde(gRandData1b, 1)
    ax_6.plot(f)
    
    # plotting with h=5
    ax_7 = fig3.add_subplot(2, 2, 3)
    g = mykde(gRandData1b, 5)
    ax_7.plot(g)
    
    # plotting with h=10
    ax_8 = fig3.add_subplot(2, 2, 4)
    h = mykde(gRandData1b, 10)
    ax_8.plot(h)
    
    # display gridlines 
    g1 = ax_5.grid(True)
    g2 = ax_6.grid(True)
    g3 = ax_7.grid(True)
    g4 = ax_8.grid(True)
    
    # Titles
    t1 = ax_5.set_title(r"Gaussian KDE")
    t2 = ax_6.set_title(r"Gaussian KDE")
    t3 = ax_7.set_title(r"Gaussian KDE")
    t4 = ax_8.set_title(r"Gaussian KDE")
    
    # LEGENDS
    l1 = mpatches.Patch(color=None, label='bandwidth=0.1')
    l2 = mpatches.Patch(color=None, label='bandwidth=1')
    l3 = mpatches.Patch(color=None, label='bandwidth=5')
    l4 = mpatches.Patch(color=None, label='bandwidth=10')
    
    ax_5.legend(handles=[l1])
    ax_6.legend(handles=[l2])
    ax_7.legend(handles=[l3])
    ax_8.legend(handles=[l4])
    
    # plotting the histogram for X -> data
    axH2 = fig4.add_subplot(2, 2, 1)
    axH2.hist(gRandData1b)
    
    plt.tight_layout()
    plt.show()
    
    # 1b ENDS