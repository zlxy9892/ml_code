# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 13:50:43 2016

@author: Administrator
"""

import nltk
nltk.download()

from nltk.book import  *

###搜索文本
#搜索单词
text1.concordance("monstrous")
text2.concordance("affection")
text3.concordance("lived")
text5.concordance("lol")

#搜索相似词
text1.similar("monstrous")

text2.similar("monstrous")

#搜索共同上下文
text2.common_contexts(["monstrous", "very"])

#词汇分布图
text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])

#自动生成文章
#text1.generate()

###计数词汇
len(text3)

sorted(set(text3))

len(set(text3))

#重复词密度
from __future__ import division
len(text3) / len(set(text3))

#关键词密度
text3.count("smote")

100 * text4.count('a') / len(text4)

def lexical_diversity(text): 
    return len(text) / len(set(text)) 
    
def percentage(count, total): 
    return 100 * count / total
    
lexical_diversity(text3)

lexical_diversity(text5)

percentage(4, 5)

percentage(text4.count('a'), len(text4))


###词链表
sent1 = ['Call', 'me', 'Ishmael', '.']
sent1

len(sent1)

lexical_diversity(sent1)

print(sent2)
print(sent3)

#连接
sent4+sent1
#追加
sent1.append("some")
print(sent1)

#索引
text4[173]

text4.index('awaken')

#切片
print(text5[16715:16735])
print(text6[1600:1625])

#索引从0开始，要注意
sent = ['word1', 'word2', 'word3', 'word4', 'word5','word6', 'word7', 'word8', 'word9', 'word10']
print(sent[0])
print(sent[9])

print(sent[10])

print(sent[5:8])
print(sent[5])
print(sent[6])
print(sent[7])

print(sent[:3])
print(text2[141525:])

sent[0] = 'First'
sent[9] = 'Last'
sent[1:9] = ['Second', 'Third']
print(sent)
sent[9]

###简单统计
#频率分布
fdist1 = FreqDist(text1)
fdist1

vocabulary1 = fdist1.keys()
vocabulary1[:50]

fdist1['whale']

fdist1.plot(50, cumulative=True)

fdist1.hapaxes()

#细粒度的选择词
V = set(text4)
long_words = [w for w in V if len(w) > 15]
sorted(long_words)

V = set(text5)
long_words = [w for w in V if len(w) > 15]
sorted(long_words)

fdist5 = FreqDist(text5)
sorted([w for w in set(text5) if len(w) > 7 and fdist5[w] > 7])

#词语搭配
from nltk.util import bigrams
list(bigrams(['more', 'is', 'said', 'than', 'done']))

text4.collocations()

text8.collocations()

###其他统计结果
[len(w) for w in text1]

fdist = FreqDist([len(w) for w in text1])
fdist
fdist.keys()

fdist.items()

fdist.max()

fdist[3]

fdist.freq(3)