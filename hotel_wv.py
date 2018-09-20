import gensim
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
del df['user_name']
del df['time']
del df['scrape_time']
df.hotel_name = 'hotel_001'
df.score[df.score < 4.0] = 0  # notice: do NOT interchange this and below lines!
df.score[df.score > 0] = 1
#df.score.apply(lambda x : int(x))
#print(df.dtypes)
#print(df.head(100))

# read in stop words list
stop_words_fname = 'stopwords.txt'
try:
    stop_words_file = codecs.open(stop_words_fname, 'r', encoding='utf')
    stop_words = stop_words_file.read().split()
    stop_words_file.close()
    #print(stop_words)
except FileNotFoundError:
    print("file name '{}' NOT found!".format(stop_words_fname))


# cut words and filter out stopwords
comment_sents = []
print('\n=====================')
for text in df.comment:
    #print("text({}): ".format(type(text)), text)
    if not isinstance(text, str):
        continue
    word_list = list(jieba.cut(text))
    filtered = [w for w in word_list if w not in stop_words]
    comment_sents.append(filtered)

# debug info
#print(comment_sents[:10])
print('total comment sentences:', len(comment_sents))

all_words = []
for sentence in comment_sents:
    all_words += sentence

theDict = set(all_words)
#print(theDict)
print(len(theDict))

n_features = 100
n_cores = 4
window_size = 10
downsampling = 1e-3

print('Training model ...')
model = gensim.models.Word2Vec(comment_sents, size=n_features, 
                                window=window_size, sample=downsampling,
                                min_count=5, workers=n_cores)

print('training loss:', model.get_latest_training_loss())
w = all_words[13]
#print(model[w])
print('------------------------')
print('word is:', w)
print(model.most_similar(w))
