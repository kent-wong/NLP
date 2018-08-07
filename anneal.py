import nltk
from random import randint

text = "doyouseethekittyseethedoggydoyoulikethekittylikethedoggy"
seg1 = "0000000000000001000000000010000000000000000100000000000"
seg2 = "0100100100100001001001000010100100010010000100010010000"
seg3 = "0000100100000011001000000110000100010000001100010000001"

# 根据分段标记对文本进行分段（分词）
def segment(text, segs):
    words = []
    last = 0
    for i in range(len(segs)):
        if segs[i] == '1':
            words.append(text[last:i+1])
            last = (i + 1)
    words.append(text[last:])
    return words

# 为分段结果打分，分数越小越好
# 综合考虑两个因素：1、‘段’的数目；2、‘段’的重复度
def evaluate(text, segs):
    words = segment(text, segs)
    lexicon_size = len(' '.join(list(set(words))))
    return len(words) + lexicon_size


def flip(segs, pos):
    return segs[:pos] + str(1 - int(segs[pos])) + segs[pos+1:]

def flip_n(segs, n):
    for i in range(n):
        segs = flip(segs, randint(0, len(segs)-1))
    return segs


# 模拟退火算法
def anneal(text, segs, iterations, cooling_rate):
    # 当前最佳分段和对应的最佳分数
    best_segs, best_score = segs, evaluate(text, segs)

    temperature = float(len(segs))
    while temperature > 0.5:
        segs = best_segs  # 一次温度迭代内，使用当前获得的最佳分段作为基础分段
        for i in range(iterations):
            n = int(round(temperature, 0))
            guess = flip_n(segs, n)
            score = evaluate(text, guess)
            if score < best_score:
                best_score, best_segs = score, guess

        temperature /= cooling_rate # 降温
        print(evaluate(text, best_segs), segment(text, best_segs))

    print()
    return segs
                

print('-----------------------')
a = anneal(text, seg1, 10000, 1.1)
print(a)
a = anneal(text, seg2, 10000, 1.1)
print(a)
a = anneal(text, seg3, 10000, 1.1)
print(a)
