
from math import  exp

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

def gradent_ascent(dataMatin,label):
    dataMat = mat(dataMatin)

