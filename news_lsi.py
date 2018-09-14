import logging
from gensim import corpora, models, similarities, matutils
from news_topics import NewsCorpus
import numpy as np
from sklearn.cluster import KMeans

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

news_corpus = NewsCorpus()
news_corpus.load()

tfidf = models.TfidfModel(news_corpus)
print(tfidf)

tfidf_corpus = tfidf[news_corpus]
print(tfidf_corpus)

lsi = models.LsiModel(tfidf_corpus, id2word=news_corpus.dictionary, num_topics=10)
print()
lsi.print_topics()

lsi_corpus = lsi[tfidf_corpus]
# save
corpora.MmCorpus.serialize('lsi_corpus.mm', lsi_corpus)

#for doc in lsi_corpus:
    #print(doc)

