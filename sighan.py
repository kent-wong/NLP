import os
import re
import jieba

class Dataset:
    def __init__(self):
        # 数据集, 每个句子占用一个列表元素
        # 一个列表元素的格式为('原始句子', '参考标注', '预测结果')
        # '预测结果'可以为空
        self._data = []

    # function:
    #       从文件读入数据集，每行一个句子
    # params:
    #       filename - 数据文件名
    #       already_cut - 是否已经进行了分词
    #       encoding - 数据文件编码
    def read(self, filename, already_cut=True, encoding='utf-8'):
        with open(filename, 'r', encoding=encoding) as f:
            print('filename:', f.name)

            for line in f.readlines():
                line = line.strip()
                if already_cut != True: #如果不是训练集，将句子直接保存到列表
                    self._data.append((line, None, None))
                    continue

                # 以下为对训练集的处理，生成的标注序列和对应的句子一样长
                sentence = line.split()
                label = []
                for w in sentence:
                    s = len(w)
                    if s == 1:
                        label.append('S')
                    else:
                        label.append('B' + 'M'*(s-2) + 'E')

                self._data.append((''.join(sentence), ''.join(label), None))

    def output(self, filename, predict=True):
        with open(filename, 'w') as f:
            for sent, gold, pred in self._data:
                cutby = pred
                if predict != True:
                    cutby = gold

                char_str = ''
                for char, tag in zip(sent, cutby):
                    char_str += char
                    if tag == 'E' or tag == 'S':
                        char_str += ' '

                f.write(char_str + '\n')


    def data(self):
        for sent in self._data:
            yield sent

    def __getitem__(self, key):
        print('key:', key)
        print(type(key))

        if isinstance(key, slice):
            start = key.start
            end = key.stop
        elif isinstance(key, int):
            start = key
            end = key + 1
        else:
            raise IndexError

        return self._data[start:end]

    def __len__(self):
        return len(self._data)

    def pprint(self, idx=None):
        if idx is None:
            start = 0
            end = len(self._data)
        else:
            start = idx
            end = idx + 1

        for sent, label, _ in self._data[start: end]:
            tag_str = ''
            char_str = ''

            for char, tag in zip(sent, label):
                tag_str += '{} '.format(tag)
                char_str += '{}'.format(char)
                if tag == 'E' or tag == 'S':
                    tag_str += '  '
                    char_str += '  '

            print(tag_str)
            print(char_str)

    def output_crfpp_format(self, filename=None):
        lines = []    
        for sent, label, _ in self._data:
            for char, tag in zip(sent, label):
                line = '{}    {}'.format(char, tag)
                lines.append(line) 
            lines.append('')
    
        if filename is not None:
            with open(filename, 'w') as f:
                f.write('\n'.join(lines))
        else:
            print('\n'.join(lines))

    def read_crfpp_result(self, filename):
        sent_str = ''
        gold_str = ''
        pred_str = ''
        with open(filename, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line == '':
                    self._data.append((sent_str, gold_str, pred_str))
                    sent_str = ''
                    gold_str = ''
                    pred_str = ''
                    continue

                char, gold, pred = line.split()
                sent_str += char
                gold_str += gold
                pred_str += pred

            if sent_str != '':
                self._data.append((sent_str, gold_str, pred_str))




#a = Dataset('pku_training.utf8')
#for sent in a.data():
#    print(sent)


#a = Dataset('pku_training.utf8')
for i in range(5):
    print()
    #print(a[i])
    
#print(a[0:5])

#a.output_crfpp_format('pku_crf_train.txt')
#a.output_crfpp_format()

#a.pprint(0)

test_data = Dataset()
#test_data.read('./sighan_corpora/gold/pku_test_gold.utf8')
test_data.read_crfpp_result('crf_out.txt')
test_data.pprint(0)
test_data.pprint(1)
test_data.pprint(2)
print('len:', len(test_data))
#test_data.output_crfpp_format('pku_test.txt')
test_data.output('pku_predict_result.txt')
