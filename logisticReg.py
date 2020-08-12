# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 16:16:32 2020

@author: senbin
"""

# two classes classfication, we put 0 as 0, others 1

import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def dataLoader(fileName):
    data = pd.read_csv(fileName)
    X = np.array(data.iloc[:,1:])
    # normalize, but can also be neglected here since the images are close
    X = X/255
    y = np.array(data.iloc[:,0])
    # two classes, we keep y=1 still 1, the others to be 0, similar to that in perceptron.py
    y[y!=1] = 0
    return X, y

class logisticReg(object):
    def __init__(self, weightSize, lr=0.001, iter=200):
        self.iter = iter
        # initialize the weights, includes w and b; and learning rate, lr
        self.weights = np.zeros(weightSize)
        self.lr = lr
    
    # use logistic func to calc the y_pred with input X
    def predict(self, X):
        # wX+b, since w and X are matrix, use np.dot() or @ to obatain a number
        # prob = np
        wX = self.weights[:-1]@X
        b = self.weights[-1]
        prob = np.exp( wX+b ) / ( 1 + np.exp(wX+b) )
        # when prob>=0.5, it gives out 0, meaning that it predicts 0 correctly
        if prob >= 0.5:
            return 1  
        return 0
    
    # train the model, updating the weights, including w and b
    def train(self, trainData, trainLabel):
        for _ in range(self.iter):
            for X, y in zip(trainData, trainLabel):
                wX = np.dot(self.weights[:-1], X)
                b = self.weights[-1]
                # w, b are updated with the same equation, but X is predefined here, b is calced later
                self.weights[:-1] += self.lr * ( X*y - np.exp(wX+b)*X ) / ( 1+np.exp(wX+b) )
                # for b, it is like X=1 in the equation above
                self.weights[-1] += self.lr * ( y - np.exp(wX+b) ) / ( 1+np.exp(wX+b) )
    
    # obtain the accuracy
    def test(self, testData, testLabel):
        count = 0
        for X, y in zip(testData, testLabel):
            # after the training, self.weights are updated, when y==y_pred, count
            y_pred = self.predict(X)
            if y == y_pred:
                count += 1
        accuracy = count / len(testLabel)
        return accuracy
    
if __name__=='__main__':
    # data and label
    trainData, trainLabel = dataLoader('data_MNIST/mnist_train.csv')
    testData, testLabel = dataLoader('data_MNIST/mnist_test.csv')
    
    # used for the train and test time statistic
    startTime=time.time()
    # initialize the perceptron class and train to update the weights
    # weightSize consists of w+b, w is the same size of x, b is bias
    weightSize = len(trainData[0]) + 1
    clf1 = logisticReg(weightSize=weightSize, lr=0.01, iter=100)
    # start training
    print('training model')
    clf1.train(trainData, trainLabel)
    # test
    print('testing model')
    accuracy = clf1.test(testData, testLabel)
    
    # for time statistic
    endTime=time.time()
    # print accuracy and time elapsed
    print('accuracy from logistic regression:', accuracy)
    print('time elapsed:', endTime-startTime)

    # use sklearn.linear_model.LogisticRegression for testing
    startTime=time.time()
    clf = LogisticRegression(random_state=0)
    clf.fit(trainData, trainLabel)
    res=clf.score(testData,testLabel)
    endTime=time.time()
    # print perceptron model results from sklearn
    print('accuracy from sklearn logistic regression:', res)
    print('time elapsed:', endTime-startTime)

