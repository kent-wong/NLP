import gensim
from gensim.models import Word2Vec
import json
import jieba
import torch
import numpy as np

# load the pretrained word vector model
model = Word2Vec.load('models/word2vec/word2vec_wx')

with open('models/WebQA/me_train.json', 'r') as f:
    data = json.load(f)

def sub_list(a, b):
    end_a = len(a) - 1

    for i, end in enumerate(b):
        if i < end_a:
            continue
        b_str = ''.join(b[i-end_a:i+1])
        a_str = ''.join(a)
        if a_str == b_str:
            return (i-end_a, i)
    return None

def extract_answers(query, data):
    result = []
    query = query.strip()
    while query[-1] == '？' or query[-1] == '?':
        query = query[:-1]
    if len(query) == 0:
        return []

    for key, value in data.items():
        answer = value['answer'][0]
        if answer.find('no_answer') == -1:
            context = value['evidence']
            tri = (query, answer, context)
            result.append(tri)

    return result

def words_to_tensor_matrix(model_wv, words, force=True):
    alist = []
    for w in words:
        try:
            v = model_wv[w]
        except KeyError:
            if force:
                return None
            else:
                v = np.zeros(model_wv.wv.vector_size, dtype=np.float32)
        v = torch.tensor(v).unsqueeze(0)
        alist.append(v)

    return torch.cat(alist, dim=0)


def load_qa():
    answers = [] # list of tuples
    for key, value in data.items():
        query = value['question']
        evidences = value['evidences']
        answers += extract_answers(query, evidences)

    print('total answers:', len(answers))
    #print(answers)

    for q, a, c in answers:
        qq = jieba.lcut(q)
        aa = jieba.lcut(a)
        cc = jieba.lcut(c)

        indexes = sub_list(aa, cc)
        if indexes is None:
            continue

        indexes_tensor = torch.tensor(indexes, dtype=torch.long)

        query_tensor = words_to_tensor_matrix(model, qq)
        if query_tensor is None:
            continue

        context_tensor = words_to_tensor_matrix(model, cc, force=False)

        #print(qq, len(qq))
        #print(query_tensor, len(query_tensor))
        #print(cc, len(cc))
        #print(context_tensor, len(context_tensor))
        #print('answer indexes:', indexes_tensor)

        #yield query_tensor, context_tensor, indexes_tensor, qq, aa, cc
        yield query_tensor, context_tensor, indexes_tensor


if __name__ == '__main__':
    for query, context, idx, q, a, c in load_qa():
        print()
        print(query)
        print(context)
        print(idx)
        print(a)
        p1 = idx[0].item()
        p2 = idx[1].item()
        aa = c[p1:p2+1]
        print(aa)

        if len(a) != len(aa):
            assert ValueError

        for i in range(len(a)):
            if a[i] != aa[i]:
                assert ValueError

