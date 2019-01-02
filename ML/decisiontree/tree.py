
from math import log
from collections import defaultdict
import numpy as np

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
            reduceFeatVec = f[:axis]   # filter row that not equal specific value
            reduceFeatVec += f[axis+1:]
            retData.append(reduceFeatVec)
    return retData

# choose best feature index
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
            prob = len(subSet) / float(len(data)) # probability with selected attribute diff values
            newEntropy += prob * calcShannonEnt(subSet) # conditional distribute with entropy
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature

# get majority class for classlist
def majorityCnt(classList):
    labelCount = defaultdict(int)
    for e in classList:
        labelCount[e] += 1
    sortedClass = sorted(labelCount.items(),key = lambda x :x[1],reverse=True)
    return sortedClass[0][0] # most class label

# create tree
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

import matplotlib.pyplot as plt
decisionNode = dict(boxstyle="sawtooth",fc="0.8")
leafNode = dict(boxstyle="round4",fc="0.8")
def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xytext=centerPt,textcoords='axs fraction',
            va='center',ha='center',bbox=nodeType,arrowprops=arrow_args)


def createPlot():
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111,frameon=False)
    plotNode('a decision node',(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode('a leaf node',(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()


def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for k in secondDict.keys():
        if type(secondDict[k]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[k])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for k in secondDict.keys():
        if type(secondDict[k]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[k])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

def retrieveTree(i):
    listOfTrees = [
            {
                'no surfacing':{0:'no',1:{ 'flippers': \
                        {0:'no',1:'yes'}}}},
                {'no surfacing':{0:'no',1:{ 'flippers': \
                        {0:{'head':{ 0:'no',1:'yes'}},1:'no'}}}}
            ]
    return listOfTrees[i]

# classify with decision tree
def classify(tree,featLabels,testVec):
    firstStr = tree.keys()[0] # first selected attribute name
    secondDict = tree[firstStr] # attribute tree
    featIndex = featLabels.index[firstStr] # selected attribute index
    for k in secondDict.keys():
        if testVec[featIndex] == k:
            if type(secondDict[k]).__name__  == 'dict':
                classlabel = classify(secondDict[k],featLabels,testVec)
            else:
                classLabel = secondDict[k]
    return classLabel

# store tree with pickle
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()

# restore tree with pickle
def grabTree(filename):
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)


if __name__ == '__main__':
    print("main")

































