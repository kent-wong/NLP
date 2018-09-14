import logging
from gensim import corpora, models, similarities, matutils
from news_topics import NewsCorpus
import numpy as np
from sklearn.cluster import KMeans

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

news_corpus = NewsCorpus()
news_corpus.load()

#print(news_corpus.all_titles)

lsi_corpus = corpora.MmCorpus('lsi_corpus.mm')
print(lsi_corpus)

X = matutils.corpus2dense(lsi_corpus, 10).T
print(X.shape)
print(X)
print(lsi_corpus[0])
print(lsi_corpus[-1])
print()

k = 12
# k 值测试，以下为对应的inertia：
# 8:  1143
# 10: 864
# 12: 672
# 16: 500
# 20: 408
kmeans = KMeans(n_clusters=k).fit(X)
XX = kmeans.transform(X)

print(kmeans.labels_.shape)
print(kmeans.labels_)
print(kmeans.cluster_centers_.shape)
#print(kmeans.cluster_centers_)
print('inertia: ', kmeans.inertia_)
print(XX)

for cluster in range(k):
    news_in_cluster = (kmeans.labels_ == cluster)
    count = 0
    print()
    print('-----------------------------------')
    print('news group: ', cluster)
    for ok, title in zip(news_in_cluster, news_corpus.all_titles):
        if ok:
            print(title)
            print()
            count += 1
            if count >= 10:
                break
