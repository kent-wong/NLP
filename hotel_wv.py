import gensim
from gensim.models import Word2Vec
import logging
import numpy as np
import pandas as pd
import jieba
import codecs
import os

fname = os.path.join('all_corpus', 'xiecheng_hotel_comments01.csv')
print(fname)

df = pd.read_csv(fname, sep=',', 
        names=['hotel_name', 'user_name', 'score', 'comment', 'time', 'scrape_time'], 
        skiprows=1)
#df = pd.read_csv(fname, sep=',')
#print(df.head())
print()

print('\n=====================')
del df['hotel_name']
del df['user_name']
del df['time']
del df['scrape_time']
#df.hotel_name = 'hotel_001'

# get rid of 'NaN'
#df.score[df.score.isnull()] = 0
df.dropna(inplace=True)

# NOTICE: do NOT interchange these two lines
df.score[df.score < 4.0] = 0
df.score[df.score > 0] = 1


# read in stop words list
stop_words_fname = 'stopwords.txt'
try:
    stop_words_file = codecs.open(stop_words_fname, 'r', encoding='utf')
    stop_words = stop_words_file.read().split()
    stop_words_file.close()
    #print(stop_words)
except FileNotFoundError:
    print("file name '{}' NOT found!".format(stop_words_fname))


model = Word2Vec.load('models/word2vec.model')

def to_comment_vector(comment):
    word_list = list(jieba.cut(comment))
    filtered = [w for w in word_list if w not in stop_words]
    a = np.array([model.wv[w] for w in filtered if w in model])
    return a.mean(axis=0)

df['vec'] = df.comment.apply(to_comment_vector)
df.dropna(inplace=True)
print(df)
