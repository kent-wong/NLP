import gensim
from gensim.models import Word2Vec
import logging
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


#model = Word2Vec.load('models/word2vec_from_weixin/word2vec_wx')

#model.save('models/word2vec.model')


model = Word2Vec.load('models/word2vec.model')

print(model.most_similar('精通'))
