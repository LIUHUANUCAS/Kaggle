# encoding=utf-8
import jieba

def loadfile(filename):
    rawdata = []
    i = 0
    with open(filename) as f:
        for line in f.readlines():
            lines = line.rstrip().split('\t')
            if i == 0 :
                i = 1
                continue
            # print lines[0],lines[-1]
            rawdata.append( (lines[0],lines[-1]) )
    return rawdata

filename= 'title.log'
rawdata = loadfile(filename)
wordcode = {}
for e in rawdata:
    seg_list = jieba.cut(e[1])
    for w in seg_list:
        wordcode[w] = 1
    # print ','.join(seg_list) 

# for k,v in wordcode.items():
    # print k,v
wordlist = wordcode.keys()
wordlist = sorted(wordlist) 
with open('wordcode.txt','w') as f:
    for w in wordlist:
       f.write(w.encode("utf8"))
       f.write('\n')
# seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
# print("Full Mode: " + "/ ".join(seg_list))  # 全模式

# seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
# print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

# seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
# print(", ".join(seg_list))

# seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
# print(", ".join(seg_list))