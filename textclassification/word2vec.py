# encoding=utf-8
import jieba
import jieba.analyse

def loaddic(filename):
    wordcodebook = {}
    i = 0
    with open(filename) as f:
        for line in f.readlines():
            lines = line.rstrip()
            wordcodebook[lines] = i
            i += 1
    return wordcodebook

filename= 'wordcode.txt'
wordcodebook = loaddic(filename)
def loadfile(filename):
    rawdata = []
    i = 0
    with open(filename) as f:
        for line in f.readlines():
            lines = line.rstrip().split('\t')
            # print lines[0],lines[-1]
            if i == 0 :
                i  =1 
                continue
            rawdata.append( (lines[0],lines[-1] ))
    return rawdata

def getvec():
    rawdata = loadfile('title.log')
    wordvec = []
    for e in rawdata:
        tags = jieba.analyse.extract_tags(e[1], topK=30, withWeight=True, allowPOS=())
        rowvec =  [0] * len(wordcodebook)
        for w in tags:
            index = wordcodebook.get(w[0])
            if index is None:
                continue
            rowvec[index] = w[1]
            # print w[1]
        rowvec = [e[0]] + rowvec
        wordvec.append(rowvec)
    return wordvec

def getvec2():
    rawdata = loadfile('title.log')
    # print rawdata[0]

    wordvec = []
    i = 0
    for e in rawdata:
        if i == 0 :
            i = 1 
            continue
        tags = jieba.analyse.extract_tags(e[1], topK=20, withWeight=True, allowPOS=())
        vec = [0] * 20
        for i,v in enumerate(tags):
            vec[i] = v[1]
       
        rowvec = [str(e[0])] + vec 
        wordvec.append(rowvec)
    return wordvec

def getvec3():
    rawdata = loadfile('title.log')
    # print rawdata[0]

    wordvec = []
    i = 0
    for e in rawdata:
        if i == 0 :
            i = 1 
            continue
        i += 1
        tags = jieba.analyse.extract_tags(e[1], topK=10)
        # print tags
        print e
        for tag in tags:
            print tag,
        print 
        
        rowvec = {k[0]:k[1] for k in tags}
        # print rowvec
        if i == 10 :
            break
        # vec = [0] * 20
        # for i,v in enumerate(tags):
        #     vec[i] = v[1]
       
        # rowvec = [str(e[0])] + vec 
        wordvec.append(rowvec)
    return wordvec

getvec3()
# wordvec = getvec2()
# with open('titlevec2.txt','w') as f:
#     # head = ['status'] + [str(i) for i in range(len(wordcodebook))]
#     head = ['status'] + ['attr_'+str(i) for i in range(20)]

#     f.write(','.join(head))
#     f.write('\n')
#     print ','.join(head)

#     for row in wordvec:
#         rowstr = ','.join([str(i) for i in row])
#         print rowstr
#         f.write(rowstr)
#         f.write('\n')

        

# seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
# print("Full Mode: " + "/ ".join(seg_list))  # 全模式

# seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
# print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

# seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
# print(", ".join(seg_list))

# seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
# print(", ".join(seg_list))