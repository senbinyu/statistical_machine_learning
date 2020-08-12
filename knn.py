# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 23:12:54 2020

@author: senbin
"""

import pandas as pd
import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier

def dataLoader(fileName):
    # read X and y for training and testing
    data = pd.read_csv(fileName)
    # transfer from pandas to np.array format, easier for the later calc
    X = np.array(data.iloc[:,1:])
    y = np.array(data.iloc[:,0])
    return X, y

class knn(object):
    # initialize the knn class, define a nearest parameter
    def __init__(self, kNearest=3):
        self.kNearest = kNearest
        
    
    # here calc the distance between X1 (each vec in the traindata) and X2 (each vec in the testData)
    def calcDist(self, X1, X2):
    # return the L1/L2 norm as the distance
    # return np.sum(np.abs(x1 - x2))
        return np.sqrt(np.sum(np.square(X1 - X2)))
    
    # use the train data to obtain the distance, and get the k nearest points to X2
    # X2 is referred to the single testdata here
    def train(self, trainData, trainLabel, X2):
        # initialize a 1d vec to store distance
        dist = np.zeros( len(trainLabel) )
        # for each vec in trainData, calc its distance to X2
        for i in range(len(trainData)):
            X1 = trainData[i]
            # store each dist in the 1d vec
            dist[i] = self.calcDist(X1, X2)
        # argsort the 1d , the k-smallest values means the k-nearest index to X2
        kNearestIndex = np.argsort(dist)[:self.kNearest]
        
        # to return a final label in the kNearest elements, the most voted is the last label
        # first, initialize a list to store the numbers of various labels
        labelList = [0] * 10
        # for each kNearestIndex, count each trainLabel
        for i in kNearestIndex:
            labelList[int(trainLabel[i])] += 1
        #return the final label which has the most counting
        finalLabel = labelList.index(max(labelList))
        return finalLabel
        
    # return the test accuracy by using the train func above. 
    # since the whole dataset is big, meaning a big matrix. 200 is default to save the time
    def test(self, trainData, trainLabel, testData, testLabel, testNum=200):
        # for each testData, we have a finalLabel calculated from train func above, and finally make a statistic
        count = 0
        for i in range(testNum):
            X2 = testData[i]
            # the realistic y2 and predicted y2
            y2 = testLabel[i]
            y2Pred = self.train(trainData, trainLabel, X2)
            if y2Pred == y2:
                count += 1
        # return the accuracy
        accuracy = count / testNum
        return accuracy
        
if __name__=='__main__':
   
    # data and label
    print('data loading')
    trainData, trainLabel = dataLoader('data_MNIST/mnist_train.csv')
    testData, testLabel = dataLoader('data_MNIST/mnist_test.csv')
    
    # here for the knn, the dataset is too big, so truncate it into a smaller one to save time
    trainData, trainLabel = trainData[:2000][:], trainLabel[:2000][:]
    testData, testLabel = testData[:200][:], testLabel[:200][:]
    
    # used for the train and test time statistic
    startTime=time.time()
    # initialize the perceptron class and train to update the weights
    clf1 = knn(kNearest=5)
    # get the kNN label and score
    print('testing model')
    accuracy = clf1.test(trainData, trainLabel, testData, testLabel)
    
    # for time statistic
    endTime=time.time()
    # print accuracy and time elapsed
    print('accuracy from knn:', accuracy)
    print('time elapsed:', endTime-startTime)

    # use sklearn.neighbors.KNeighborsClassifier for testing
    startTime=time.time()
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(trainData, trainLabel)
    # .score() func can give out the mean accuracy
    res=clf.score(testData,testLabel)
    endTime=time.time()
    # print perceptron model results from sklearn
    print('accuracy from sklearn knn:', res)
    print('time elapsed:', endTime-startTime)