from numpy import *

def setOfWords2Vec(vocabList,inputSet):
    vec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            vec[vocabList.index(word)] = 1
        else:
            print("word:%s not in vocabulary" % word)
    return vec

def createVocabList(dataSet):
    vocabSet = set([])
    for doc in dataSet:
        vocabSet = vocabSet | set(doc)
    return list(vocabSet)

# count occurency of word in document
def bagOfWords2VecMN(vocabList,inputSet):
    vec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            vec[vocabList.index(word)] += 1
    return vec

def textParse(text):
    import re
    listword = re.split(r'\W',text)
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
        #wordstringdata = [re.split(r'\W',e.rstrip()) for e in f.readlines()]
        wordstringdata = [e.rstrip() for e in f.readlines()]

    #print(wordstringdata)

    doclist = []
    label = []
    labelMap = {'ham':0,'spam':1}
    labellist = []
    for doc in wordstringdata:
        #wordlist = re.split(r'\W*',doc) 
        wordlist = re.split(r'\W',doc) 
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

    testsize = int(len(train_array)*0.80)

    test_array = train_array[testsize:]
    test_label = label[testsize:]
    train_array_ = train_array[:testsize]
    train_label_ = label[:testsize]

    from sklearn.naive_bayes import GaussianNB
    from sklearn.naive_bayes import MultinomialNB 
    #clf = GaussianNB()
    clf = MultinomialNB()
    clf.fit(train_array_,train_label_)
    test_res = clf.predict(test_array)
    diff = [(u == v) for u,v in zip(test_label,test_res)]

    errcount = sum([e==True for e in diff])

    errcount = float(errcount)/len(test_array)
    print('right rate:%f'% errcount )



smsclassification()

