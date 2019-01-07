
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

# stochastic gradient ascent
def stochastic_gradient_ascent(dataMatrix,classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmod(sum(dataMatrix[i]*weights))
        err = classLabels[i] - h
        weights += alpha * err * dataMatrix[i]
    return weights

def stochastic_gradient_ascent_iter(dataMatrix,classLabels,numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = [e for e in range(m)]
        for i in range(m):
            alpha = 4 / (1.0 +i +j ) + 0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmod( sum(dataMatrix[randIndex]*weights) )
            err = classLabels[randIndex] - h
            weights += alpha *err * dataMatrix[randIndex]
            del(dataIndex[randIndex])

    return weights


def classifyVector(inX,weights):
    prob = sigmod(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else :
        return 0.0

def colicTest():
    train_data = []
    train_label = []
    with open('horseColicTraining.txt') as f:
        for e in f.readlines():
            line_array = e.rstrip().split('\t')
            vec = [float(x) for x in line_array[:20]]
            label = float(line_array[21])
            train_data.append(vec)
            train_label.append(label)

    #weights = stochastic_gradient_ascent(array(train_data),train_label)
    weights = stochastic_gradient_ascent_iter(array(train_data),train_label,500*2)
    print(shape(weights))
    errcount = 0
    test_len = 0
    with open('./horseColicTest.txt') as f:
        for e in f.readlines():
            test_len +=1 
            line_array = e.rstrip().split('\t')
            test_vec = [float(x) for x in line_array[:20] ]
            test_label = float(line_array[21])
            test_array = array(test_vec)
            class_label = classifyVector(test_array,weights)
            if int(class_label) != int(test_label):
                errcount += 1

    errrate= errcount/ float(test_len)
    print('errrate:%f'%(errcount/float(test_len)))
    return errrate




















