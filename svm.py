# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:03:36 2020

@author: senbin
"""

import time
import numpy as np
import pandas as pd
import random
from sklearn.svm import SVC

def dataLoader(fileName):
    # use MNIST data here, and to consider it as two classes
    # separate the whole data into data X and label y
    data = pd.read_csv(fileName)
    X = np.array(data.iloc[:,1:])
    # normalize, but can also be neglected here since the images are close
    X = X/255
    y = np.array(data.iloc[:,0])
    
    # we keep 1 to be 1, and others to be -1
    y[y != 1] = -1
    
    return X, y

class svm(object):
    # initialize the parameters
    def __init__(self, trainData, trainLabel, sigma=10, C=200, toler=0.001, iter=40):
        # transfer parameters, not change in class svm
        self.sigma = sigma
        self.C = C
        self.toler = toler
        self.iter = iter
        self.sampleNum, self.featureNum = np.shape(trainData)
        self.trainData = trainData
        self.trainLabel = trainLabel
        # calced parameters later
        self.kernel = self.calcKernel()
        self.alpha = [0] * self.sampleNum
        self.b = 0
        self.E = -self.trainLabel # according to the book, initial: alpha=0, b=0, gxi=0, E=gxi-yi
        self.supportVecInd = []
      
    # calc a gaussian kernel for the smo in training
    def calcKernel(self):
        # initialize an array to store the kernel X*X, with the shape of sampleNum
        kernel = [ [0 for i in range(self.sampleNum) ] for j in range(self.sampleNum )]
        #np.zeros((self.sampleNum, self.sampleNum))
        
        # take each sample as feature, to calc the gaussian kernel
        for i, X1 in enumerate(self.trainData):
            # since kernel[i][j] = kernel[j][i], calc once is enough
            for j in range(i, self.sampleNum):
                X2 = trainData[j]
                # temp  to store the ||X1-X2||^2
                temp = (X1 - X2) @ (X1 - X2).T
                temp = np.exp(-1*temp / (2*self.sigma**2)) # use gassuian kernel
                kernel[i][j] = temp
                kernel[j][i] = temp
        
        return kernel
    
    # judge if ith sample satify the KKT condition
    def satisfyKKT(self, i):
        yi, gxi = trainLabel[i], self.calcGxi(i)
        
        # conditions according to the book
        # print(gxi)
        if (abs(self.alpha[i]) < self.toler) and (yi*gxi >= 1):
            return True
        elif (abs(self.alpha[i] - self.C) < self.toler) and (yi*gxi <= 1):
            return True
        elif (self.alpha[i] > -self.toler) and (self.alpha[i]<(self.C+self.toler)) and (abs(yi*gxi-1)<self.toler):
            return True
        
        # when not satisfy the conditions above, return False
        return False
    
    # calc gxi for the judgement of satifying KKT conditions
    def calcGxi(self, i):
        gxi = 0
        index = [i for i, alpha in enumerate(self.alpha) if alpha != 0]
        # gxi=sum(alpha_j * y_j * kernel[j][i]) + b
        for ind in index:
            gxi += self.alpha[ind] * self.trainLabel[ind] * self.kernel[ind][i] 
        gxi += self.b
        return gxi 
    
    # Ei = gxi - yi, used in calc alphaNew
    def calcEi(self, i):
        return self.calcGxi(i) - self.trainLabel[i]
        
    # SMO, use E1 to find the E2 and alpha2
    def calcAlphaj(self, E1, i):
        # initialize maxE1_E2 and maxInd to be -1, easier for the later calc
        maxE1_E2, maxInd = -1, -1
        nonZeroEInd = [i for i, Ei in enumerate(self.E) if Ei!=0]
        for j in nonZeroEInd:
            E2_temp = self.calcEi(j)
            if abs(E1 - E2_temp) > maxE1_E2:
                maxE1_E2 = abs(E1 - E2_temp)
                E2 = E2_temp
                maxInd = j
        # if all of them are zeros, choose one maxIndex different from i, and calc E2 later
        if maxInd == -1:
            maxInd = i
            while maxInd == i:
                maxInd = int(random.uniform(0, self.sampleNum))
            E2 = self.calcEi(maxInd)
        
        return E2, maxInd
    
    def train(self):
        iterStep, marker = 0, 1
        # when iteration step is not enough and has large residence
        while iterStep<self.iter and marker>0:
            # counting the steps and re-initialize the marker
            iterStep, marker = iterStep+1, 0
            
            # for each sample, calc alpha1, alpha2
            for i in range(self.sampleNum):
                # when the sample does not satisfy the KKT, calc E1, then to calc E2
                if not self.satisfyKKT(i):
                    E1 = self.calcEi(i)
                    E2, j = self.calcAlphaj(E1, i)
                    
                    # list the kernels for the calc of alphaNew
                    kernel11 = self.kernel[i][i]
                    kernel12 = self.kernel[i][j]
                    kernel21 = self.kernel[j][i]
                    kernel22 = self.kernel[j][j]
                    
                    # list of labels and L, H to calc alphaNew
                    y1, y2 = self.trainLabel[i], self.trainLabel[j]
                    alpha_1Old, alpha_2Old = self.alpha[i], self.alpha[j]
                    # according to the book equation, define the L, H, which are used for alphaNew
                    if y1 != y2:
                        L = max(0, alpha_2Old - alpha_1Old)
                        H = min(self.C, self.C + alpha_2Old - alpha_1Old)
                    else:
                        L = max(0, alpha_2Old + alpha_1Old - self.C)
                        H = min(self.C, alpha_2Old + alpha_1Old)
                    
                    # since L<=alpha2New<=H, if L=H, alpha2New can not be updated, so continue
                    if L == H:
                        continue
                    # otherwise, update the alpha2New, if in the range L--H, unchanged
                    alpha_2New = alpha_2Old + y2 * (E1-E2) / (kernel11 + kernel22 - 2*kernel12)
                    if alpha_2New < L:
                        alpha_2New = L
                    elif alpha_2New > H:
                        alpha_2New = H
                    
                    # then use alpha2_New calc alpha_1New
                    alpha_1New = alpha_1Old + y1*y2*(alpha_2Old - alpha_2New)
                    
                    # calc bNew according to the book LiHang
                    b_1New = -1 * E1 - y1 * kernel11 * (alpha_1New - alpha_1Old) \
                            - y2 * kernel21 * (alpha_2New - alpha_2Old) + self.b
                    b_2New = -1 * E2 - y1 * kernel12 * (alpha_1New- alpha_1Old) \
                            - y2 * kernel22 * (alpha_2New - alpha_2Old) + self.b
                    if (alpha_1New > 0) and (alpha_1New < self.C):
                        bNew = b_1New
                    elif (alpha_2New > 0) and (alpha_2New < self.C):
                        bNew = b_2New
                    else:
                        bNew = (b_1New + b_2New) / 2
                    
                    # write the updated value into the matrix
                    self.alpha[i], self.alpha[j], self.b = alpha_1New, alpha_2New, bNew
                    self.E[i], self.E[j] = self.calcEi(i), self.calcEi(j)
                    # if the change of alpha_2New is also small, consider it's not changed
                    if abs(alpha_2New - alpha_2Old) >= 0.0001:
                        marker += 1
            # use print to detect the iterations      
            # print('iteration step: %d, i: %d, pairs changed: %d', %(iterStep, i, marker) )
            
        # find the supportVector index if alpha>0
        for i in range(self.sampleNum):
            if self.alpha[i] > 0:
                self.supportVecInd.append(i)
    
    def predict(self, Xi):
        # only each supportVector plays a role on the X
        res = 0
        for i in self.supportVecInd:
            #print(i)
            temp = self.calcSingleKernel(self.trainData[i,:], Xi)
            res += self.alpha[i] * self.trainLabel[i] * temp
        # add the bias later
        res += self.b
        return np.sign(res)
    
    def calcSingleKernel(self, X1, X2):
        res = (X1 - X2) @ (X1 - X2).T
        singleKernel = np.exp(-res / (2*self.sigma**2))
        return singleKernel
    
    def test(self, testData, testLabel):
        # counting the correct numbers
        count = 0
        for data, label in zip(testData, testLabel):
            pred = self.predict(data)
            # print('pred', pred)
            if label == pred:
                count += 1
        accuracy =  count/len(testLabel)
        return accuracy

if __name__=='__main__':
   
    # data and label
    trainData, trainLabel = dataLoader('data_MNIST/mnist_train.csv')
    testData, testLabel = dataLoader('data_MNIST/mnist_test.csv')
    
    # truncate the dataset to save time
    trainData, trainLabel = trainData[:500][:], trainLabel[:500][:]
    testData, testLabel = testData[:100][:], testLabel[:100][:]
    
    # used for the train and test time statistic
    startTime=time.time()
    # initialize the svm class and train to update the weights
    clf1 = svm(trainData, trainLabel, iter=50)
    # start training
    print('training model')
    clf1.train()
    # test
    print('testing model')
    accuracy = clf1.test(testData, testLabel)
    
    # for time statistic
    endTime=time.time()
    # print accuracy and time elapsed
    print('accuracy from svm:', accuracy)
    print('time elapsed:', endTime-startTime)

    # use sklearn.svm.SVM for testing
    startTime=time.time()
    clf = SVC()
    clf.fit(trainData, trainLabel)
    res=clf.score(testData,testLabel)
    endTime=time.time()
    # print perceptron model results from sklearn
    print('accuracy from sklearn perceptron:', res)
    print('time elapsed:', endTime-startTime)

