
from math import  exp
from numpy import *

def loadSet():
    data = []
    label = []
    with open('./testSet.txt') as f:
        for e in f.readlines():
            array_ = e.rstrip().split()
            data.append([1.0,float(array_[0]),float(array_[1])])
            label.append(int(array_[2]))

    return data,label

# sigmod function
def sigmod(x):
    return 1.0/(1 + exp(-x))

# gradient ascent
def gradient_ascent(dataMatin,label):
    dataMat = mat(dataMatin)
    labelMat = mat(label).transpose()
    m,n = shape(dataMat)
    alpha = 0.001
    max_cycles = 500
    weights = ones((n,1))
    for k in range (max_cycles):
        h = sigmod(dataMat*weights)
        err = (labelMat - h)
        weights += alpha * dataMat.transpose() *err
    return weights

def stochastic_gradient_ascent(dataMatrix,classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmod(sum(dataMatrix[i]*weights))
        err = classLabels[i] - h
        weights += alpha * err * dataMatrix[i]
    return weights
