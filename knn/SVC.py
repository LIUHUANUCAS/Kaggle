__author__ = 'LIUHUAN'
import csv
import pandas as pd
from sklearn import svm,metrics
from  numpy import  *
import pickle
from sklearn.externals import joblib
def load_cvs_data(filename):
    reader = csv.reader(file(filename,'rb'))
    data=[]
    print type(reader)
    # return [],[]
    target =[]
    count = 0
    for line in reader:
        if count == 0 :
            count = 1
            continue
        y = int(line[0])
        target.append([y])
        data.append([int(e) for e in line[1:]])
        # print type(line)
    print array(target).shape
    print array(data).shape
    return array(data),array(target)

def load_csv_data_pandas(filename):
    dframe = pd.read_csv(filename)
    # X=dframe[1:-1,1:-1]
    y = dframe['label']
    # print y.shape
    # print dframe.shape
    X = dframe.iloc[:,1:]
    return array(X),array(y)
    # print X
    # print X.shape
    # y=dframe[1:-1,0]
    # print dframe
def load_csv_test_data_pandas(filename):
    dframe = pd.read_csv(filename)
    print 'begin load test data'
    
    y=dframe['label'][10467:]
    x=dframe.iloc[10467:,1:]
    return array(x),array(y)

def train():
    filename = 'train.csv'
    # filename = 'tt.csv'
    traindata,label = load_csv_data_pandas(filename)
    # traindata,label = load_cvs_data(filename)
    print len(traindata)
    print 'load finished...'
    clf = svm.SVC(gamma=0.001,C=100.)
    print 'svm finish...'
    clf.fit(traindata,label)
    print 'fit finished...'
    joblib.dump(clf,'digit.pkl')
    print 'dump finished....'
    # pickle.dumps(clf)
    # load_cvs_data_pandas(filename)
def test():
    filename = 'train.csv'
    testdata,label= load_csv_test_data_pandas(filename)
    # testdata,label= load_cvs_data_pandas(filename)
    print 'load data finished...'
    print testdata.shape
    print label.shape
    clf = joblib.load('digit.pkl')
    print 'load model finished ....'
    
    # predict test data
    predicted = clf.predict(testdata)
    # num = 0
    # error=0
    
    # sum(e for e in label==predicted if e is True)
    # for i,j in zip(label,predicted):
    #     if i ==j :
    #         num+=1
    #     else:
    #         error+=1
           
    # print num,len(label),double(num)/len(label)
    # print error,len(label),double(error)/len(label)
    
    print 'predict data finished...'
    # print 'report %s->\n%s\n'  % (clf,metrics.classification_report(label,predicted))
    with open('res.pkl','wb') as f:
        pickle.dump(predicted,f)
        
def test_digit_handwrite():
    filename = 'test.csv'
    print 'being load ...'
    df = pd.read_csv(filename)
    # print df.iloc[0]
    X = df.iloc[:,:]
    X=array(X)
    clf = joblib.load('digit.pkl')
    print 'begin predict'
    print X.shape
    # print X[0]
    predicted=[]
    for x in X:
        label = clf.predict(x)
        predicted.append(label)
    print 'save results...'
    with open('results.pkl','wb') as f:
        pickle.dump(predicted,f)
    # print 'report %s->\n%s\n'  % (clf,metrics.classification_report(label,predicted))
    
    
#train()
test_digit_handwrite()
