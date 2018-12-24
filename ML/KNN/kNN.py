import numpy as np
import operator

def createDataSet():
    group = array([
        [1.0,1.1],[1.0,1.0],[0,0],[0,0.1],
    ])
    labels = ['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat **2 
    sqDistances = sqDiffMat.sum(axis =1 )
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k) :
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    label_ = {'didntLike':0,'largeDoses':1,'smallDoses':2}
    data = []
    label = []
    with open(filename) as f:
        for line in f.readlines():
            line = line.rstrip()
            e = line.split('\t')
            money = float(e[0])
            play_time = float(e[1])
            ice = float(e[2])
            data.append([money,play_time,ice])
            label.append(e[-1])
            #label.append(label_[e[-1]])

    return np.array(data),np.array(label)

#data,label = file2matrix('./datingTestSet.txt')
#print len(data),len(label)
#
#print data[0][0] + 1
#print label[0]
