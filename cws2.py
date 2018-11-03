# !/usr/bin/python3
import argparse
import sys
import json
import time

class Weights(dict): # 管理平均感知器的权重
    def __init__(self):
        self._values = dict()
        self._last_step = dict()
        self._step = 0

        self._usedict = False
        self._words = dict()

        self._acc = dict()

    def _new_value(self, key):
        dstep = self._step-self._last_step[key]
        value = self._values[key]

        self._acc[key] += dstep*value
        self._last_step[key] = self._step
        return value

    def update_all(self):
        for key in self._values:
            self._new_value(key)

    def update_weights(self,key,delta): # 更新权重
        if key not in self._values : 
            self._values[key]=0
            self._acc[key]=0
            self._last_step[key]=self._step
        else :
            self._new_value(key)

        self._values[key]+=delta

    def average(self): # 平均
        self._backup=dict(self._values)
        for k,v in self._acc.items():
            self._values[k]=self._acc[k]/self._step

    def unaverage(self): 
        self._values=dict(self._backup)
        self._backup.clear()

    def save(self,filename):
        json.dump({k:v for k,v in self._values.items() if v!=0.0},
                open(filename,'w', encoding='utf-8'),
                ensure_ascii=False,indent=1)

    def load(self,filename):
        self._values.update(json.load(open(filename, encoding='utf-8')))
        self._last_step = None
    
    def get_value(self, key, default):
        if key not in self._values:
            return default
        if self._last_step is None:
            return self._values[key]
        return self._new_value(key)

    # 融入词典
    def merge_dict(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            self._usedict = True

            for word in f.readlines():
                word = word.strip()
                if len(word) <= 1:
                    continue

                # 正向
                make_word = ''
                for w in word:
                    make_word += w
                    if make_word not in self._words:
                        self._words[make_word] = 0
                self._words[word] = 1

                
    def max_match(self, sub):
        matched = 0
        for i in range(len(sub)):
            w = sub[:i+1]
            if w in self._words:
                if self._words[w] == 1:
                    matched = len(w)
            else:
                break
        return matched


class CWS :
    def __init__(self, words_list=None):
        self.weights=Weights()
        # 载入词典
        if words_list:
            self.weights.merge_dict(words_list)
            #print('dict loaded')
            #print(self.weights._words)
            #matched = self.weights.max_match('前景是光明的')
            #print(matched)


    def gen_features(self,x): # 枚举得到每个字的特征向量
        for i in range(len(x)):
            #left3=x[i-3] if i-3 >=0 else '#'
            left2 = x[i-2] if i-2 >=0 else '#'
            left1 = x[i-1] if i-1 >=0 else '#'
            mid = x[i]
            right1 = x[i+1] if i+1<len(x) else '#'
            right2 = x[i+2] if i+2<len(x) else '#'
            #right3=x[i+3] if i+3<len(x) else '#'
            triple = left1 + mid + right1

            features = ['1'+mid, '2'+left1, '3'+right1,
                        '4'+left2+left1, '5'+left1+mid, '6'+mid+right1, '7'+right1+right2,
                        '0_T' + triple]

            yield features

    def update(self,x,y,delta): # 更新权重
        for i, features in zip(range(len(x)), self.gen_features(x)):
            for feature in features :
                self.weights.update_weights(str(y[i])+feature, delta)
        for i in range(len(x)-1):
            self.weights.update_weights(str(y[i])+':'+str(y[i+1]), delta)

    def decode(self, x): # 类似隐马模型的动态规划解码算法
        # 类似隐马模型中的转移概率
        transitions = [ [self.weights.get_value(str(i)+':'+str(j),0) for j in range(4)]
                for i in range(4) ]

        # 类似隐马模型中的发射概率
        emissions = [ [sum(self.weights.get_value(str(tag)+feature,0) for feature in features) 
            for tag in range(4) ] for features in self.gen_features(x)]


        # 类似隐马模型中的前向概率
        alphas = [[[e,None] for e in emissions[0]]]
        for i in range(len(x)-1) :
            alphas.append([max([alphas[i][j][0]+transitions[j][k]+emissions[i+1][k],j]
                                        for j in range(4))
                                        for k in range(4)])
        # 根据alphas中的“指针”得到最优序列
        alpha = max([alphas[-1][j],j] for j in range(4))
        i = len(x)
        tags = []
        while i :
            tags.append(alpha[1])
            i-=1
            alpha=alphas[i][alpha[1]]

        labels = list(reversed(tags))

        # 使用词典建议解码
        if self.weights._usedict:
            #print('origin:   ', labels) # debug
            i = 0
            while i < len(x):
                matched = self.weights.max_match(x[i:])
                if matched <= 3:
                    i += 1
                    continue

                # 将几个相邻的词组成一个词，提高准确率
                if (labels[i] == 0 or labels[i] == 3) and (labels[i+matched-1] == 2 or labels[i+matched-1] == 3):
                    labels[i] = 0
                    labels[i+matched-1] = 2
                    for idx in range(i+1, i+matched-1):
                        labels[idx] = 1
                # 将长词切断，提高召回率
                #elif labels[i] == 0:
                #    for ii in range(i+1, i+matched):
                #        if labels[ii] != 1:
                #            break
                #    else:
                #        labels[i+matched-1] = 2
                #        if labels[i+matched] == 1:
                #            labels[i+matched] = 0
                #        else:
                #            labels[i+matched] = 3
                #elif labels[i+matched-1] == 2:
                    #for ii in range(i, i+matched-1):
                    #    if labels[ii] != 1:
                    #        break
                    #else:
                    #    labels[i] = 0
                    #    if labels[i-1] == 1:
                    #        labels[i-1] = 2
                    #    else:
                    #        labels[i-1] = 3

                i += matched

            #print('suggested:', labels) # debug


        return labels

    def verbose(self, x):
        d = {0: 'B:', 1: 'M:', 2: 'E:', 3: 'S:'}
        emissions = [ [sum(self.weights.get_value(str(tag)+feature,0) for feature in features) 
            for tag in range(4) ] for features in self.gen_features(x)]

        for tag in range(4):
            print(d[tag], end=' ')
            for i in range(len(x)):
                value = emissions[i][tag]
                print('{:8.2f}'.format(value), end=' ')
            print()

        print('-')
        for tag in range(4):
            print(d[tag], end=' ')
            for w in x:
                value = self.weights.get_value(str(tag) + '8B_' + w + '2', 0)
                print('{:8.2f}'.format(value), end=' ')
            print()

def load_example(words): # 词数组，得到x，y
    y=[]
    for word in words :
        if len(word)==1 : y.append(3)
        else : y.extend([0]+[1]*(len(word)-2)+[2])
    return ''.join(words),y

def dump_example(x,y) : # 根据x，y得到词数组
    cache=''
    words=[]
    for i in range(len(x)) :
        cache+=x[i]
        if y[i]==2 or y[i]==3 :
            words.append(cache)
            cache=''
    if cache : words.append(cache)
    return words

class Evaluator : # 评价
    def __init__(self):
        self.std,self.rst,self.cor=0,0,0
        self.start_time=time.time()

    def _gen_set(self,words):
        offset=0
        word_set=set()
        for word in words:
            word_set.add((offset,word))
            offset+=len(word)
        return word_set

    def __call__(self,std,rst): # 根据答案std和结果rst进行统计
        std, rst = self._gen_set(std),self._gen_set(rst)
        self.std += len(std)
        self.rst += len(rst)
        self.cor += len(std&rst)

    def report(self):
        precision = self.cor/self.rst if self.rst else 0
        recall = self.cor/self.std if self.std else 0
        f1 = 2*precision*recall/(precision+recall) if precision+recall!=0 else 0
        print("历时: %.2f秒 答案词数: %i 结果词数: %i 正确词数: %i  准确率：%.4f 召回率：%.4f F值: %.4f"
                %(time.time()-self.start_time,self.std,self.rst,self.cor, precision, recall, f1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--iteration', type=int, default=5, help='')
    parser.add_argument('--train', type=str, help='')
    parser.add_argument('--test', type=str, help='')
    parser.add_argument('--dev', type=str, help='')
    parser.add_argument('--predict', type=str, help='')
    parser.add_argument('--result', type=str, help='')
    parser.add_argument('--model', type=str, help='')
    parser.add_argument('--dict', type=str, help='')
    parser.add_argument('--verbose', help='', action='store_true')
    parser.add_argument('--score', type=str, help='')
    parser.add_argument('--ref', type=str, help='')
    parser.add_argument('--stats', help='show statistics info about model', action='store_true')
    parser.add_argument('--reduce', type=str, help='specify new reduced model name')
    parser.add_argument('--below', type=int, help='')
    args = parser.parse_args()

    # 训练
    if args.train: 
        cws=CWS(words_list=args.dict)
        for i in range(args.iteration):
            print('\n**第 %i 次迭代:'%(i+1))
            #sys.stdout.flush()
            evaluator=Evaluator()
            lines = 0
            for sent in open(args.train, encoding='utf-8'):
                sent = sent.strip()
                if sent == '':
                    continue
                x,y=load_example(sent.split())
                z=cws.decode(x)
                evaluator(dump_example(x,y),dump_example(x,z))
                cws.weights._step+=1
                if z!=y :
                    cws.update(x,y,1)
                    cws.update(x,z,-1)

                # debug
                lines += 1
                if lines % 2000 == 0:
                    print('trained sentences {}'.format(lines))

            evaluator.report()
            cws.weights.update_all()
            #cws.weights.average()

            if args.dev :
                evaluator=Evaluator()
                for sent in open(args.dev, encoding='utf-8') :
                    sent = sent.strip()
                    if sent == '':
                        continue
                    x,y=load_example(sent.split())
                    z=cws.decode(x)
                    evaluator(dump_example(x, y), dump_example(x,z))
                evaluator.report()
            #cws.weights.unaverage()

        cws.weights.average() # 最后再进行平均
        cws.weights.save(args.model)


    # 使用有正确答案的语料测试
    if args.test : 
        cws = CWS(words_list=args.dict)
        cws.weights.load(args.model)
        evaluator = Evaluator()
        for sent in open(args.test, encoding='utf-8') :
            sent = sent.strip()
            if sent == '':
                continue
            x,y = load_example(sent.split())
            z = cws.decode(x)
            evaluator(dump_example(x, y), dump_example(x, z))
        evaluator.report()

    if args.score and args.ref: 
        evaluator = Evaluator()
        f_score = open(args.score, encoding='utf-8')
        f_ref = open(args.ref, encoding='utf-8')
        counter = 0
        while True:
            try:
                line1 = next(f_ref)
                line2 = next(f_score)
                if args.verbose:
                    print('{}:'.format(counter))
                    counter += 1
                    print(line1)
                    print(line2)

                x1, y1 = load_example(line1.split())
                x2, y2 = load_example(line2.split())
                evaluator(dump_example(x1, y1), dump_example(x2, y2))
            except StopIteration:
                break
        evaluator.report()

    if args.stats:
        if args.model:
            cws_stats = CWS(words_list=args.dict)
            cws_stats.weights.load(args.model)

            total_weights = len(cws_stats.weights._values)
            weight_class = (0.001, 0.01, 0.1, 0.5, 1, 10, 100)
            class_stats = {}
            for n in weight_class:
                class_stats[n] = 0
            class_stats[-1] = 0

            for k, v in cws_stats.weights._values.items():
                v = abs(v)
                for n in weight_class:
                    if v < n:
                        class_stats[n] += 1
                        break
                else:
                    class_stats[-1] += 1


            print('*statistics info*:')
            print('total weights:', total_weights)
            print('weights distribution:')
            others = 0
            prev = -1
            for k, v in sorted(class_stats.items()):
                if k == -1:
                    others = v
                else:
                    if prev == -1:
                        prev = 0
                    print('{:6}  - {:6}:   {} ({:.2f}%)'.format(prev, k, v, v*100/(total_weights+0.000001)))
                prev = k
            print('{:6}  - {:6}:   {} ({:.2f}%)'.format(weight_class[-1], '', others, others*100/(total_weights+0.000001)))

            if cws_stats.weights._usedict:
                print('user dict info:')
                for k, v in cws_stats.weights._words.items():
                    print(k, v)
        else:
            print('MUST specify `--model` option')

    if args.reduce:
        if not args.model:
            print('MUST specify `--model` option')
            sys.exit(1)
        if args.below is None:
            print('MUST specify `--below` option')
            sys.exit(1)
        if args.below <= 0:
            print('`below` must be > 0')
            sys.exit(1)

        cws_reduced = CWS()
        cws_reduced.weights.load(args.model)

        total_weights = len(cws_reduced.weights._values)

        temp = dict(cws_reduced.weights._values)
        #weights_list = sorted(cws_reduced.weights._values.items(), key=lambda x: abs(x[1]))
        for k, v in temp.items():
            if abs(v) < args.below:
                del cws_reduced.weights._values[k]

        cws_reduced.weights.save(args.reduce)

        print('number of weights before reduce:', total_weights)
        print('after:', len(cws_reduced.weights._values))
        sys.exit(0)

    # 对未分词的句子输出分词结果
    if args.model and (not args.train and not args.test and not args.stats) : 
        cws = CWS(words_list=args.dict)
        cws.weights.load(args.model)
        instream = open(args.predict, encoding='utf-8') if args.predict else sys.stdin
        outstream = open(args.result,'w', encoding='utf-8') if args.result else sys.stdout
        for sent in instream:
            sent = sent.strip()
            if sent == '':
                continue
            x, y = load_example(sent.split())
            z = cws.decode(x)
            print(' / '.join(dump_example(x, z)), file=outstream)
            if args.verbose:
                cws.verbose(sent)
