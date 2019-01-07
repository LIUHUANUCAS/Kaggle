from numpy import *
def colicTest():
    train_data = []
    train_label = []
    with open('horseColicTraining.txt') as f:
        for e in f.readlines():
            line_array = e.rstrip().split('\t')
            vec = [float(x) for x in line_array[:20]]
            label = float(line_array[21])
            train_data.append(vec)
            train_label.append(label)

    test_len = 0
    test_array = []
    test_label = []
    with open('./horseColicTest.txt') as f:
        for e in f.readlines():
            line_array = e.rstrip().split('\t')
            test_vec = [float(x) for x in line_array[:20] ]
            test_label.append(float(line_array[21]))
            test_array.append(array(test_vec))


    from sklearn import linear_model
    from sklearn import neighbors 
    logistic = linear_model.LogisticRegression(solver='lbfgs',max_iter=10000)
    logistic.fit(train_data,train_label)
    test_res = logistic.predict(test_array)
    diff = [u==v for u,v in zip(test_label,test_res)]
    errate = sum([e== True for e in diff]) / float(len(diff))
    print('errorate:%f'%errate)
    
    #print('score:%r' % logistic.fit(train_data,train_label).score(test_array,test_label) )


colicTest()

