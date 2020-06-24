# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 23:54:24 2020

@author: sundesh raj
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

class NaiveBayes(object):
    
    def __init__(self, setNum=1, iterItem=2, roc=0, scatter=0, mean1=[], mean2=[], std1=[], std2=[]):
        self.setNum = setNum
        self.iterItem = iterItem
        self.roc = roc
        self.scatter = scatter
        self.mean1 = mean1
        self.mean2 = mean2
        self.std1 = std1
        self.std2 = std2
        self.accuracyDict = {}
        self.item = 1
        self.dataLabel_1 = [500]
        self.dataLabel_0 = [500]
        
        if self.iterItem == 2:
            self.item = 1
            self.dataLabel_1 = [500]
            self.dataLabel_0 = [500]
        elif self.iterItem == 3:
            self.item = 6
            self.dataLabel_1 = [10, 20, 50, 100, 300, 500]
            self.dataLabel_0 = [10, 20, 50, 100, 300, 500]
        elif self.iterItem == 4:
            self.item = 1
            self.dataLabel_1 = [300]
            self.dataLabel_0 = [700]
            
        for x in range(self.item):
            trainingData1 = np.random.multivariate_normal(self.mean1, self.std1, self.dataLabel_0[x])
            trData1DF = pd.DataFrame(trainingData1)
            trData1DF.columns = ["Col1", "Col2"]
            trData1DF["label"] = 0
            
            trainingData2 = np.random.multivariate_normal(self.mean2, self.std2, self.dataLabel_1[x])
            trData2DF = pd.DataFrame(trainingData2)
            trData2DF.columns = ["Col1", "Col2"]
            trData2DF["label"] = 1
            
            trainDF = trData1DF.append(trData2DF, ignore_index = True, sort = False)
            trainDF = trainDF.sample(frac=1).reset_index(drop = True)
            
            X = trainDF[["Col1", "Col2"]]
            Y = trainDF[["label"]]
            
            testData1 = np.random.multivariate_normal(self.mean1, self.std1, 500)
            testData1DF = pd.DataFrame(testData1)
            testData1DF.columns = ["Col1", "Col2"]
            testData1DF["label"] = str(0)
            
            testData2 = np.random.multivariate_normal(self.mean2, self.std2, 500)
            testData2DF = pd.DataFrame(testData2)
            testData2DF.columns = ["Col1", "Col2"]
            testData2DF["label"] = str(1)
            
            testDF = testData1DF.append(testData2DF, ignore_index = True, sort = False)
            testDF = testDF.sample(frac=1).reset_index(drop = True)
            
            X_Test = testDF
            Y_Test = pd.DataFrame()
            
            NaiveBayes.myNB(self, X, Y, X_Test, Y_Test)
            
        if self.iterItem == 3:
                print(self.accuracyDict)
                lts = sorted(self.accuracyDict.items())
                x,y = zip(*lts)
                plt.plot(x,y)
                plt.xlabel("Num of Samples")
                plt.ylabel("Testing Accuracies")
                plt.show()
    
    def calculateMean (self, inputData):
        m = inputData.mean(axis=0)
        return m
    
    def calculateSD (self, inputData):
        stdDev = inputData.std(axis=0)
        return stdDev
    
    def divideData (self, inputData):
        label0 = pd.DataFrame()
        label1 = pd.DataFrame()
        
        for i, row in inputData.iterrows():
            if row["label"] == 1:
                label1 = label1.append(pd.Series([row["Col1"], row["Col2"]]), ignore_index = True)
            else:
                label0 = label0.append(pd.Series([row["Col1"], row["Col2"]]), ignore_index = True)
        
        label0.columns = ["Col1", "Col2"]
        label1.columns = ["Col1", "Col2"]
        
        return label0, label1
    
    def getLikelihood (self, val, mean, stdDeviation):
        valLikelihood = scipy.stats.norm(mean, stdDeviation).pdf(val)
        return valLikelihood
    
    def getPriorLabels (self, inputData):
        cLabel1 = 0
        cLabel0 = 0
        for i, row in inputData.iterrows():
            if row["label"] == 1:
                cLabel1 += 1
            else:
                cLabel0 += 1
        return cLabel0, cLabel1
    
    def getLabelLikelihood (self, val, prior, mean, stdDeviation):
        val_Likelihood = prior
        for i in val:
            if i == val[0]:
                val_Likelihood *= NaiveBayes.getLikelihood(self, i, mean.loc["Col1"], stdDeviation.loc["Col1"])
            else:
                val_Likelihood *= NaiveBayes.getLikelihood(self, i, mean.loc["Col2"], stdDeviation.loc["Col2"])
        return val_Likelihood
    
    def plotScatter (self, accDF):
        for i, row in accDF.iterrows():
            if row["predLabel"] == "1":
                plt.scatter
    
    
    def myNB (self, X, Y, X_Test, Y_Test):
        
        
        train = X.join(Y)
        
        trainLabel0, trainLabel1 = NaiveBayes.divideData (self, train)
        
        trainLabel0_mean = NaiveBayes.calculateMean (self, trainLabel0)
        trainLabel0_std = NaiveBayes.calculateSD (self, trainLabel0)
        
        trainLabel1_mean = NaiveBayes.calculateMean (self, trainLabel1)
        trainLabel1_std = NaiveBayes.calculateSD (self, trainLabel1)
        
        priorLabel0, prioLabel1 = NaiveBayes.getPriorLabels (self, train)
        
        # PERFORMING PREDICTION
        for i, row in X_Test.iterrows():
            value = [row["Col1"], row["Col2"]]
            
            l1Likelihood = NaiveBayes.getLabelLikelihood (self, value, prioLabel1, trainLabel1_mean, trainLabel1_std)
            l0Likelihood = NaiveBayes.getLabelLikelihood (self, value, priorLabel0, trainLabel0_mean, trainLabel0_std)
            
            l1Prob = l1Likelihood/(l1Likelihood+l0Likelihood)
            l0Prob = l0Likelihood/(l1Likelihood+l0Likelihood)
            
            if l1Prob > l0Prob:
                label = str(1)
            else:
                label = str(0)
            
            Y_Test = Y_Test.append(pd.Series([row["Col1"], row["Col2"], l1Prob, l0Prob, label]), ignore_index = True)
        Y_Test.columns = ["Col1", "Col2", "l1Posterior", "l0Posterior", "predLabel"]
        print()
        print("Printing the posterior probability and predicted labels for the dataset")
        print(Y_Test)
        
        # ACCURACY
        cMatch = 0
        accDF = pd.DataFrame()
        
        for (i1, row1), (i2, row2) in zip(Y_Test.iterrows(), X_Test.iterrows()):
            if row1["predLabel"] == row2["label"]:
                cMatch += 1
                accDF = accDF.append(pd.Series([row1["Col1"], row1["Col2"], row1["predLabel"], row2["label"]]), ignore_index = True)
            else:
                continue
        accDF.columns = ["Col1", "Col2", "predLabel", "actualLabel"]
        print()
        print("Correct Prediction : {}".format(cMatch))
        print()
        acc = (cMatch)/(len(Y_Test.index))*100
        print("Accuracy : {}".format(acc))
        print("Error : {}".format(100-acc))
        
        if self.iterItem == 3:
            self.accuracyDict.update({int(X.shape[0]/2) : acc})
        
        # CALCULATE PRECISION RECALL AND CONFUSION MATRIX
        truePositive = 0
        trueNegative = 0
        for i, row in accDF.iterrows():
            if row["predLabel"] == "1":
                truePositive += 1
            else:
                trueNegative += 1
        
        falsePositive = 0
        falseNegative = 0
        for (i1, row1), (i2, row2) in zip(Y_Test.iterrows(), X_Test.iterrows()):
            if row1["predLabel"] == "1" and row2["label"] == "0":
                falsePositive += 1
            if row1["predLabel"] == "0" and row2["label"] == "1":
                falseNegative += 1
                
        print()
        print("##########################################################")
        print("##################CONFUSION MATRIX########################")
        print("##########################################################")
              
        cfmDict = {('actual class','positive'):[truePositive, falseNegative], ('actual class', 'negative'):[falsePositive, trueNegative]}
        index = [['predLabel','predLabel'],['positive', 'negative']]
        confMatrix = pd.DataFrame(cfmDict, index=index)
        print()
        print(confMatrix)
        
        print()
        print("##########################################################")
        print("-----------------------PRECISION--------------------------")
        print("##########################################################")
        prec = truePositive/(truePositive+falsePositive)
        print(prec)
        print()
        print("##########################################################")
        print("-------------------------RECALL---------------------------")
        print("##########################################################")
        recall = truePositive/(truePositive+falseNegative)
        print(recall)
        
        # Scatter plot
        try:
            if (self.iterItem == 2 or self.iterItem == 4) and self.scatter == 1:
                for i,row in accDF.iterrows():
                    if row["predLabel"] == "1":
                        plt.scatter(row["Col1"], row["Col2"], color="blue")
                    else:
                        plt.scatter(row["Col1"], row["Col2"], color="red")
                plt.xlabel("xplots")
                plt.ylabel("yplots")
                plt.show()
        except (IndexError):
            pass
        

        # Plot ROC
        try:
            if (self.iterItem == 2 or self.iterItem == 4) and self.roc == 1:
                ROCDF = pd.DataFrame()
                for (i1,row1), (i2,row2) in zip(Y_Test.iterrows(), X_Test.iterrows()):
                    ROCDF = ROCDF.append(pd.Series([row1["Col1"], row1["Col2"], row2["label"], row1["l1Posterior"], row1["l0Posterior"]]), ignore_index = True)
                ROCDF.columns = ["Col1", "Col2", "ActualClass", "L1Posterior", "L0Posterior"]
                ROCDF = ROCDF.sort_values(by="L1Posterior", ascending = False)
                ROCDF["PredLabel"] = str(0)
                    
                ROC_R_List = []
                ROC_R_List.append((0.0,0.0))
                    
                for i, row in ROCDF.iterrows():
                    t = row["L1Posterior"]
                    for i1, row1 in ROCDF.iterrows():
                        if float(row1["L1Posterior"]) >= float(t):
                            ROCDF.at[i1, 'PredLabel'] = str(1)
                        else:
                            continue
                            
                    true_positive = 0
                    true_negative = 0
                    false_positive = 0
                    false_negative = 0
                        
                    for i2, row2 in ROCDF.iterrows():
                        if row2["PredLabel"] == "1" and row2["ActualClass"] == "1":
                            true_positive += 1
                        elif row2["PredLabel"] == "1" and row2["ActualClass"] == "0":
                            false_positive += 1
                        elif row2["PredLabel"] == "0" and row2["ActualClass"] == "0":
                            true_negative += 1
                        elif row2["PredLabel"] == "0" and row2["ActualClass"] == "1":
                            false_negative += 1
                        
                    tPR = true_positive/(true_positive+false_negative)
                    fPR = false_positive/(true_negative+false_positive)
                        
                    ROC_R_List.append((fPR, tPR))
                
                # PLOTTING THE ROC CURVE
                print("##########################################################")
                print("-------------------------ROC---------------------------")
                print("##########################################################")
                p1, q1 = zip(*ROC_R_List)
                plt.plot(p1, q1)
                plt.xlabel("FPR")
                plt.ylabel("TPR")
                plt.show()
                    
                # AUC
                AUC_List = []
                ROC_R_List_1 = list(ROC_R_List)
                ROC_R_List_1.remove((1.0,1.0))
                    
                ROC_R_List_2 = list(ROC_R_List)
                ROC_R_List_2.remove((0.0,0.0))
                    
                for r, s in zip(ROC_R_List_1, ROC_R_List_2):
                    b = s[0] - r[0]
                    length = s[1] - 0
                    RECT_Area = length*b
                    AUC_List.append(RECT_Area)
                    
                AUC = 0.0
                for i in AUC_List:
                    AUC = AUC + float(i)
                print("AUC ------------> {}".format(AUC))
                
        except(IndexError):
            pass