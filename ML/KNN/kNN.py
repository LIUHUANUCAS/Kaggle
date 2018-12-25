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
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat **2 
    sqDistances = sqDiffMat.sum(axis =1 )
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k) :
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 
    #sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    sortedClassCount = sorted(classCount.items(),key=lambda x:x[1],reverse=True)
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

def autoNorm(data):
    min_vals = data.min(0)
    max_vals = data.max(0)
    ranges = max_vals - min_vals
    norm_data = np.zeros(np.shape(data))
    m = data.shape[0]
    norm_data = data - np.tile(min_vals,(m,1))
    norm_data = norm_data / np.tile(ranges,(m,1))
    return norm_data,ranges,min_vals

def normTestData(data,ranges,minvals):
    return (data - minvals) / ranges

def datingClassTest():
    data,label = file2matrix('./datingTestSet.txt')
    norm_data,ranges,minvals = autoNorm(data)
    print('norma_data:',norm_data.shape,'ranges:',ranges,'minvals:',minvals)

    testsize = 100
    k = 7
    r = 0
    err = 0
    classify_data = norm_data[testsize:,:]
    classify_label = label[testsize:]
    for i in range(int(testsize)):
        test_row = norm_data[i,:]
        test_label = classify0(test_row,classify_data,classify_label,k)
        real_label = label[i]
        print('[%d],test:%s,label:%s' % (i,test_label,real_label))
        if real_label == test_label:
            r += 1
        else :
            err += 1
    print('right:%d,r_ratio:%f,err:%d,err_%f'% (r,r*1.0/testsize,err,err*1.0/testsize))

if __name__ == '__main__':
    testdata = [30000,0.5,0.5]
    print (testdata)
    k = 5
    data,label = file2matrix('./datingTestSet.txt')
    norm_data,ranges,minvals = autoNorm(data)
    normtestdata = normTestData(testdata,ranges,minvals)
    test_label = classify0(normtestdata,norm_data,label,k)
    print(testdata,test_label)




#data,label = file2matrix('./datingTestSet.txt')
#print len(data),len(label)
#
#print data[0][0] + 1
#print label[0]
