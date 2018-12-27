
from math import log
from collections import defaultdict

# calculate shannon entropy
def calcShannonEnt(dataSet):
    dataLen = len(dataSet)
    labelCount = defaultdict(int)
    for f in dataSet:
        label = f[-1]
        labelCount[label] += 1
    ent = 0.0 
    for k in labelCount:
        prob = float(labelCount[k]) / dataLen
        ent -= prob *log(prob,2)

    return ent

def createDataSet():
    data = [
            [1,1,'yes'],
            [1,1,'yes'],
            [1,0,'no'],
            [0,1,'no'],
            [0,1,'no']
            ]
    label = ['no surfacing','flippers']
    return data,label

# split dataset in axis=axis on value
def splitDataSet(data,axis,value):
    retData = []
    for f in data:
        if f[axis] == value:
            reduceFeatVec = f[:axis]
            reduceFeatVec += f[axis+1:]
            retData.append(reduceFeatVec)
    return retData

def chooseBestFeatureToSplit(data):
    featLen = len(data[0]) - 1
    baseEntropy = calcShannonEnt(data)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(featLen):
        featList = [row[i] for row in data]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for v in uniqueVals:
            subSet = splitDataSet(data,i,v)
            prob = len(subSet) / float(len(data))
            newEntropy += prob * calcShannonEnt(subSet)
            infoGain = baseEntropy - newEntropy
            if (infoGain > bestInfoGain):
                bestInfoGain = infoGain
                bestFeature = i

    return bestFeature

def majorityCnt(classList):
    labelCount = defalutdict(int)
    for e in classList:
        labelCount[e] += 1
    sortedClass = sorted(labelCount.items(),key = lambda x :x[1],reverse=True)
    return sortedClass[0][0] # most class label

def createTree(dataSet,label):
    classList = [e[-1] for e in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1 :
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestLabel = label[bestFeat]
    myTree = {bestLabel:{}}
    del(label[bestFeat])
    featValues = [e[bestFeat] for e in dataSet]
    uniqVals = set(featValues)
    for v in uniqVals:
        subLabel = label[:]
        splitdata = splitDataSet(dataSet,bestFeat,v)
        myTree[bestLabel][v] = createTree(splitdata,subLabel)

    return myTree


