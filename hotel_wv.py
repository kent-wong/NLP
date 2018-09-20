import gensim
from gensim.models import Word2Vec
import logging
import numpy as np
import pandas as pd
import jieba
import codecs
import os
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

fname = os.path.join('all_corpus', 'xiecheng_hotel_comments01.csv')
print(fname)

df = pd.read_csv(fname, sep=',', 
        names=['hotel_name', 'user_name', 'score', 'comment', 'time', 'scrape_time'], 
        skiprows=1)

fname = os.path.join('all_corpus', 'xiecheng_hotel_comments02.csv')
df2 = pd.read_csv(fname, sep=',', 
        names=['hotel_name', 'user_name', 'score', 'comment', 'time', 'scrape_time'], 
        skiprows=1)

df = pd.concat([df, df2], ignore_index=True)

print('\n=====================')
del df['hotel_name']
del df['user_name']
del df['time']
del df['scrape_time']

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

X = np.array(df.vec.tolist())
y = np.array(df.score.tolist())
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25)

forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest = forest.fit(X_train, y_train)
a = confusion_matrix(y_train, forest.predict(X_train))
print(a)

print("test result:")
a = confusion_matrix(y_test, forest.predict(X_test))
print(a)
