# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 10:55:05 2020

@author: senbin
"""

# the aim here is to predict the y when we have X, the question becomes to obtain the P(y=ck|x), 
# P(y=ck|x)= P(y=ck)*P(X|y=ck) / sum( P(y=ck) P(X|y=ck) ) The denominator is a full propability
# X has many features, which can also be dismembered, here we make it to be 1 and 0

import time
import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB

def dataLoader(fileName):
    # read X and y for training and testing
    data = pd.read_csv(fileName)
    # transfer from pandas to np.array format, easier for the later calc
    X = np.array(data.iloc[:,1:])
    # transfer the feature into 1 when it is >128, otherwise, it is 0
    X[X<128] = 0; X[X>=128] = 1
    y = np.array(data.iloc[:,0])
    return X, y

class naiveBayes(object):
    def __init__(self, featureNum=784, classNum=10):
        # extract the featureNum from X and classNum from y
        self.featureNum = featureNum
        self.classNum = classNum
        
    def train(self, trainData, trianLabel):
        # calc py and px_y from the train data directly
        # first initialize them, note 2 in px_y denotes the 0 and 1 two sides
        # concretely, it is P(X=x0|y=ck), P(X=x1|y=ck)
        py = np.zeros(self.classNum)
        px_y = np.zeros((self.featureNum, 2, self.classNum))
        # calc py when y==ck, count them together to get the probability
        for i in range(self.classNum):
            # Note: to keep py not equals to 0, meanwhile denominator + classNum
            py[i] = (np.sum(trainLabel==i) + 1) / len(trainLabel + self.classNum)
        # turn it into log for the calc convinence
        py = np.log(py)
        
        # calc px_y, for each ck, it is (P(X|y=ck)), and sum all the ck together to have P(X|y)
        # for each y, calc the numbers of px_y first
        for i, ck in enumerate(trianLabel):
            X = trainData[i]
            # get the X and y=ck, for each X[j] in the feature vec, count the numbers and store into px_y
            for j in range(self.featureNum):
                # count each element into P(X=x0|y=ck), P(X=x1|y=ck)
                px_y[j] [X[j]] [ck] += 1
        # calc the numbers above, then calc the probability
        # for each y=ck, calc the P(X=x0|y=ck) and P(X=x1|y=ck)
        # the real one can be calced by P(X=x0|y=ck) / sum(P(X=x0|y=ck)+P(X=x1|y=ck))
        
        for ck in range(self.classNum):
            for j in range(self.featureNum):
                px_x0_y = px_y[j][0][ck]
                px_x1_y = px_y[j][1][ck]
                
                px_y[j][0][ck] = np.log( (px_x0_y+1) / (px_x0_y + px_x1_y+2) ) 
                px_y[j][1][ck] = np.log( (px_x1_y+1) / (px_x0_y + px_x1_y+2) )
        
        
        # similar to the above prob calc of py, avoid 0, +1 in the numerator; in the denominator, + each feature num, which is 2
        #px_y = np.log( (px_y+1) / ( np.sum(px_y, axis=1)+2 ) )
        
        return py, px_y
        
    # py is P(y=ck), px_y is P(X|y=ck) which are calc from original data above
    def predict(self, py, px_y, X):
        # initialize a list to store the probability, final p
        p = [0] * self.classNum
        for i in range(self.classNum):
            # for each y=ck, counting its prob from P(X|y=ck)
            s = 0
            for j in range(self.featureNum):
                s += px_y[j][X[j]] [i]
            # for each y=ck, since it is log, + P(y=ck) to P(X|y=ck)
            p[i] = s + py[i]
        # return the index (also the label) which has the highest P(y=ck|X)
        return p.index(max(p))
        
    # use the testData to obtain accuracy when use naiveBayes
    def test(self, trainData, trainLabel, testData, testLabel):
        # to obtain the matrix of prob first
        py, px_y = self.train(trainData, trainLabel) 
        # to count the accuracy
        count = 0
        for X, y in zip(testData, testLabel):
            # for each X, use the predict func to get y_pred
            y_pred = self.predict(py, px_y, X)
            if y == y_pred:
                count += 1
        accuracy = count / len(testLabel)
        return accuracy

if __name__=='__main__':
   
    # data and label
    print('data loading')
    trainData, trainLabel = dataLoader('data_MNIST/mnist_train.csv')
    testData, testLabel = dataLoader('data_MNIST/mnist_test.csv')
    
    # truncate can also be used here, to turn dataset into a smaller one for saving time
    #trainData, trainLabel = trainData[:2000][:], trainLabel[:2000][:]
    # testData, testLabel = testData[:200][:], testLabel[:200][:]
    
    # used for the train and test time statistic
    startTime=time.time()
    # initialize the perceptron class and train to update the weights
    nb = naiveBayes(featureNum=trainData.shape[1], classNum=10)
    # get the score
    print('testing model')
    accuracy = nb.test(trainData, trainLabel, testData, testLabel)
    
    # for time statistic
    endTime=time.time()
    # print accuracy and time elapsed
    print('accuracy from naiveBayes:', accuracy)
    print('time elapsed:', endTime-startTime)

    # use sklearn.naive_bayes.BernoulliNB for testing
    startTime=time.time()
    bnb = BernoulliNB()
    bnb.fit(trainData, trainLabel)
    # .score() func can give out the mean accuracy
    res=bnb.score(testData,testLabel)
    endTime=time.time()
    # print perceptron model results from sklearn
    print('accuracy from sklearn naiveBayes:', res)
    print('time elapsed:', endTime-startTime)
