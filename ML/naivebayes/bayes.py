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


# count occurency of word in document
def bagOfWords2VecMN(vocabList,inputSet):
    vec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            vec[vocabList.index(word)] += 1
    return vec

def textParse(text):
    import re
    listword = re.split(r'\W*',text)
    return [w.lower() for w in listword if len(w) > 2]

def spamTest():
    doclist = []
    label = []
    fulltext = []
    for i in range(1,26):
        wordlist = textParse(open('./email/spam/%d.txt'%i).read())
        doclist.append(wordlist)
        fulltext.extend(wordlist)
        label.append(1)
        wordlist = textParse(open('./email/ham/%d.txt'%i).read())
        doclist.append(wordlist)
        fulltext.extend(wordlist)
        label.append(0)
    
    # get vocabulary for all doc
    vocablist = createVocabList(doclist)

    trainingSet = range(50)
    testSet = []
    uniqmap = {}
    for i in range(50):

        randindex = int(random.uniform(0,len(trainingSet)))
        uniqmap[randindex] = 1
        if len(uniqmap) >= 10 :
            break
        # testSet.append(trainingSet[randindex])
        # del(trainingSet[randindex])
    for k in uniqmap:
        testSet.append(trainingSet[k])

    word2vec = bagOfWords2VecMN
    trainMat = []
    trainClass = []
    for docindex in trainingSet:
        if docindex in uniqmap:# skip test data
            continue
        trainMat.append(word2vec(vocablist,doclist[docindex]))
        trainClass.append(label[docindex])
    
    # train NB
    p0v ,p1v,pSpam = trainNB0(array(trainMat),array(trainClass))

    errcount = 0
    # test NB
    for docindex in testSet:
        wordvec = word2vec(vocablist,doclist[docindex])
        classlabel = classifyNB(array(wordvec),p0v,p1v,pSpam)
        if classlabel != label[docindex]:
            errcount += 1
            print(doclist[docindex])
    
    print('err rate:',float(errcount)/ len(testSet))

        
def smsclassification():
    wordstringdata = []
    
    import re
    with open('./SMSSpamCollection.txt') as f:
        #wordstringdata = [re.split(r'\W*',e.rstrip()) for e in f.readlines()]
        wordstringdata = [e.rstrip() for e in f.readlines()]

    #print(wordstringdata)

    doclist = []
    label = []
    labelMap = {'ham':0,'spam':1}
    labellist = []
    for doc in wordstringdata:
        wordlist = re.split(r'\W*',doc) 
        wordlist = [e.lower() for e in wordlist if len(e) > 2]
        doclist.append(wordlist[1:])
        labellist.append(wordlist[0])

    #print(doclist)
    #return
    vocablist = createVocabList(doclist)
    train_array = []

    for i,doc in enumerate(doclist):
        train_array.append(setOfWords2Vec(vocablist,doc))
        label.append(labelMap[labellist[i]])

    testsize = int(len(train_array)*0.85)

    test_array = train_array[testsize:]
    test_label = label[testsize:]
    train_array_ = train_array[:testsize]
    train_label = label[:testsize]

    #train NB
    p0v,p1v,pSpam = trainNB0(array(train_array_),array(train_label))
    
    #test 
    errcount = 0
    for i,d in enumerate(test_array):
        classlabel = classifyNB(array(d),p0v,p1v,pSpam)
        if classlabel != test_label[i]:
            errcount += 1 

    errcount = float(errcount)/len(test_array)
    print('err rate:%f'% errcount )



smsclassification()

