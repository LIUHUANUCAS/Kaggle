__author__ = 'LIUHUAN'
from numpy import *
import pickle
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
def load_data():
	 digit = datasets.load_digits()
	 X = digit.data[:-1]
	 y=digit.target[:-1]
	 print y.shape,X.shape
	 return array(X),array(y),digit.data[-1],digit.target[-1]
def load_trian_test_data():
	trainname = 'train.csv'
	testname = 'test.csv'
	df1 = pd.read_csv(trainname)
	df2 = pd.read_csv(testname)
	train = df1.iloc[:,1:]
	label = df1['label']
	test = df2.iloc[:,:]
	print test.shape,train.shape,label.shape
	return array(train),array(label),array(test)
def train_test():
	train,y,test = load_trian_test_data()
        print test.shape,train.shape

	knnclassifier = KNeighborsClassifier(n_neighbors=10)
	knnclassifier.fit(train,y)
        py=[]
        for e in test:
            label = knnclassifier.predict(e)
            py.append(label)

	with open('knn.pkl','wb') as f:
		pickle.dump(py,f)
	
def train():
	train,label,px,py= load_data()
	neigh = KNeighborsClassifier(n_neighbors=5)
	neigh.fit(train,label)
	print neigh.predict(array([px]))
	print py

# train_test()
def writeResult2CSVFile(filename):
	with open(filename,'rb') as f:
		res = pickle.load(f)
	with open('svc.txt','w') as f2:
		f2.write('ImageId\t')
		f2.write('Label\n')
		i=1
		for line in res:
			f2.write(str(i)+'\t')
			# f2.write(str(line[0])+'\n')
			f2.write(str(line[0])+'\n')
			# print line
			i+=1
			# print line

if __name__=='__main__':
	
    filename ='results.pkl'
    # filename ='knn.pkl'

    writeResult2CSVFile(filename)
    # train_test()	
	
