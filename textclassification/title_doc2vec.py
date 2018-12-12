# encoding=utf-8
import jieba
import jieba.analyse
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import numpy as np
import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')

class Feature :
    def init(self,label,flen,orig_sentence,cutwordlist ):
        self.label = label
        self.flen = flen
        self.sentence = orig_sentence
        self.cutwordlist = cutwordlist
    
def loadfile(filename):
    rawdata = []
    i = 0
    with open(filename) as f:
        for line in f.readlines():
            lines = line.rstrip().split('\t')
            # print lines[0],lines[-1]
            if i == 0 :
                i  = 1 
                continue
            e =(lines[0],lines[-1])
            rawdata.append( e )
            # rawdata.append( (lines[0],lines[-1] ))
    return rawdata

seta = u"。？！，、；：“”‘’（ ）《 》〈 〉【 】『 』「 」﹃ ﹄〔 〕…—～﹏￥"
stop_word_list_set = set([ e for e in seta])
print(seta)
def getstoplen(cut_word_list):
    """get stop list len"""
    return sum([1 for e in cut_word_list if e in stop_word_list_set])

def getsentencelen(sentence):
    return len(sentence)

def getvec():
    rawdata = loadfile('title.log')
    wordvec = []
    for e in rawdata:
        # print len(e[1])
        cutwordslist = jieba.cut(e[1].rstrip())
        tmplist = [x for x in cutwordslist]
        # print len(tmplist)
        e1 = (e[0],len(e[1]),getstoplen(tmplist) )
        wordvec.append( ( e1, [x for x in tmplist] ) )
    return wordvec

cutwordlist = getvec()
# print cutwordlist
for e in cutwordlist[1][1]:
    print (e in stop_word_list_set)
wordlist = [e[1] for e in cutwordlist]
print (stop_word_list_set,'，')
# print '》'.encode('utf8') in seta 
# for e in stop_word_list_set:
#     print e
# print u'》' in stop_word_list_set
# for e in wordlist:
#     print e
size = 50
# print wordlist
model = Word2Vec(wordlist,size=size,window=5,min_count=1,workers=4)


w1 = wordlist[0]
f = open('new1.csv','w')
print >> f,','.join(['label'] + ['sentence_len','stopwordcount'] +['attr_'+ str(i) for i in range(size)])
for tdata in cutwordlist:
    label = tdata[0][0]
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
    print >>f,','.join(['pass' if str(label) == '2' else 'reject'] + [str(tdata[0][1]),str(tdata[0][2]) ] + ['%.2f'%e for e in resarray])
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