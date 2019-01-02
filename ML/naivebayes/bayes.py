from numpy import *

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for doc in dataSet:
        vocabSet = vocabSet | set(doc)
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
    vec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            vec[vocabList.index(word)] = 1
        else:
            print("word:%s not in vocabulary" % word)
    return vec

# caculate the probility of every word of every class 
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    prob_abusive = sum(trainCategory)/ float(numTrainDocs)
    prob_0_num = ones(numWords)
    prob_1_num = ones(numWords)
    prob_0_denom = 2.0 # zero 
    prob_1_denom = 2.0 # zeor
    for i in range(numTrainDocs):
        if trainCategory[i] == 1 :
            prob_1_num += trainMatrix[i]
            prob_1_denom += sum(trainMatrix[i])
        else :
            prob_0_num += trainMatrix[i]
            prob_0_denom += sum(trainMatrix[i])
    prob_1_vect = log(prob_1_num / prob_1_denom)## log to prevent underflow
    prob_0_vect = log(prob_0_num / prob_0_denom)## log to prevent underflow
    return prob_0_vect,prob_1_vect,prob_abusive

def classifyNB(vec2Classify,p_0_vec,p_1_vec,pclass):
    p1 = sum(vec2Classify * p_1_vec) + log(pclass)
    p0 = sum(vec2Classify * p_0_vec) + log(1.0- pclass)
    if p1> p0:
        return 1
    else :
        return 0

def testingNB():
    data ,label = loadDataSet()
    vocabList = createVocabList(data)
    trainMat = []
    for doc in data:
        trainMat.append(setOfWords2Vec(vocabList,doc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(label))
    testEntry = ['love','my','dalmation']
    thisdoc = array(setOfWords2Vec(vocabList,testEntry))
    print(testEntry,'classified as',classifyNB(thisdoc,p0V,p1V,pAb))
    testEntry = ['stupid','garbage']
    thisdoc = array(setOfWords2Vec(vocabList,testEntry))
    print(testEntry,'classified as',classifyNB(thisdoc,p0V,p1V,pAb))
