# 基于分类的断句器
# 对有可能是句子结束标志的标点符号(例如.!?)进行分类：
# 分类：1、是句子结束标志；2、不是句子结束标志

import nltk

sents = nltk.corpus.treebank_raw.sents()
print(type(sents))
print(sents)

tokens = []
boundaries = set() # 存入语料中的句子结束位置，用于监督学习
offset = 0
for sent in sents:
    tokens += sent
    offset += len(sent)
    boundaries.add(offset-1)

# 构造特征向量
def punct_features(tokens, i):
    return {
            'next-word-cap': tokens[i+1][0].isupper() if i+1 < len(tokens) else False,
            'prevword': tokens[i-1].lower(),
            'punct': tokens[i],
            'prevword-is-one-char': len(tokens[i-1]) == 1
            }

featuresets = [(punct_features(tokens, i), (i in boundaries))
        for i in range(1, len(tokens)-1)
        if tokens[i] in '.?!'
        ]

print('-----------------------')
#print(tokens)
print(len(featuresets))

size = int(len(featuresets)*0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
cls = nltk.NaiveBayesClassifier.train(train_set)
a = nltk.classify.accuracy(cls, test_set)
print('accuracy:', a)

print('-----------------------')
print(tokens[:40])
for j in range(100):
    if tokens[j] in '.!?':
        a = cls.classify(punct_features(tokens, j))
        print(a)
        if a == True:
            print(j)

# 使用训练好的分类器进行断句
def segment_sentences(classifier, words):
    start = 0
    sents = []
    for i, w in enumerate(words):
        if w in '.?!' and classifier.classify(punct_features(words, i)) == True:
            sents.append(words[start:i+1])
            start = (i + 1)

    if start < len(words):
        sents.append(words[start:])

    return sents

a = segment_sentences(cls, tokens)
for s in a:
    print(' '.join(s))
