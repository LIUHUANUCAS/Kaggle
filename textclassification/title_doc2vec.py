# encoding=utf-8
import jieba
import jieba.analyse
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import numpy as np

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
        cutwordslist = jieba.cut(e[1],cut_all=False)
        wordvec.append( ( e[0],[x for x in cutwordslist]) )
    return wordvec
cutwordlist = getvec()
wordlist = [e[1] for e in cutwordlist]
# for e in wordlist:
#     print e
model = Word2Vec(wordlist,size=100,window=5,min_count=1,workers=4)

w1 = wordlist[0]
size = 100
f = open('new1.csv','w')
print >> f,','.join(['label'] + ['attr_'+ str(i) for i in range(size)])
for tdata in cutwordlist:
    label = tdata[0]
    doc = tdata[1]
    resarray = np.zeros((size,))
    # print resarray.shape
    # break
    for w in doc:
        # print w
        resarray += np.array(model.wv[w])
        # print model.wv[w].shape
        # break
        # print model.wv[w]
    print >>f,','.join([str(label)] + [str(e) for e in resarray])
f.close()
# print resarray
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