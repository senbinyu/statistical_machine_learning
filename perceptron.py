# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:03:36 2020

@author: senbin
"""

import time
import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron

def dataLoader(fileName):
    # use MNIST data here, and to consider it as two classes
    # separate the whole data into data X and label y
    data = pd.read_csv(fileName)
    X = np.array(data.iloc[:,1:])
    # normalize, but can also be neglected here since the images are close
    X = X/255
    y = np.array(data.iloc[:,0])
    # separate them to two classes,first we test 0-4 is -1, 5-9 is 1, accuracy around 0.81
    # but later we found that if we can make 0 to -1, the others to be 1, accuracy is increased to 0.98
    y[y==0] = -1
    y[y>0] = 1
    '''
    # initialize data and label
    data, label = [], []
    # use with open, it will close the file after the usage
    with open(fileName,'r') as f:
        for line in f.readlines(): 
            lineSplit = line.strip().split(',')  # use ',' to split the data
            data.append([int(num)/255 for num in lineSplit[1:]]) 
            if int(lineSplit[0])<5:
                label.append(-1)
            else:
                label.append(1)
    # return data and label
    X = np.array(data)
    y = np.array(label)
    '''
    return X, y
                
class perceptron(object):
    # initialize the weights, size from the weightSize
    def __init__(self, weightSize, lr=0.001, iter=10):
        self.iter = iter  # iteration numbers
        self.weights = np.zeros(weightSize)  #initialize weights
        self.lr = lr # learning rate
    
    # use predict to calculate each forward result, y=wx+b, w=weights[:-1], b=weights[-1], 
    def predict(self, X):
        # initialize the pred for storing the predicted labels
        # w and X are np.dot(), also the same with using of @
        pred = self.weights[:-1] @ X + self.weights[-1]
        # if it is <0, then make it -1, otherwise it keeps 1, no need to change
        if pred<0:
                pred = -1
        else:
                pred = 1
        return pred
    
    # read the train dataset and update the weights
    def train(self, trainData, trainLabel):
        for _ in range(self.iter):
            # each Xi is extracted to predict the label yi^, and use yi, yi^ to update weights
            for data, label in zip(trainData, trainLabel):
                pred = self.predict(data)
                # w=w+lr*(y-y^)*x, b=b+lr*(y-y^)
                if -label*pred >= 0:
                    self.weights[:-1] += self.lr * label * data
                    self.weights[-1] += self.lr * label
    
    # test the trained model
    def test(self, testData, testLabel):
        # count for the whole accuracy
        count = 0
        for data, label in zip(testData, testLabel):
            pred = self.predict(data)
            if label == pred:
                count += 1
        accuracy = count/len(testLabel)
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
    clf1 = perceptron(weightSize=weightSize, lr=0.01, iter=50)
    # start training
    print('training model')
    clf1.train(trainData, trainLabel)
    # test
    print('testing model')
    accuracy = clf1.test(testData, testLabel)
    
    # for time statistic
    endTime=time.time()
    # print accuracy and time elapsed
    print('accuracy from perceptron:', accuracy)
    print('time elapsed:', endTime-startTime)

    # use sklearn.linear_model.Perceptron for testing
    startTime=time.time()
    clf = Perceptron(tol=1e-3, random_state=0)
    clf.fit(trainData, trainLabel)
    res=clf.score(testData,testLabel)
    endTime=time.time()
    # print perceptron model results from sklearn
    print('accuracy from sklearn perceptron:', res)
    print('time elapsed:', endTime-startTime)