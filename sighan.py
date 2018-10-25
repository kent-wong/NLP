import os
import re
import jieba

class Dataset:
    # 数据集, 每个句子占用一个列表元素
    # 一个列表元素的格式为('原始句子', '对应的标注')
    # 标注方法：使用4标注BMES
    _data = []

    # params:
    #       filename - 数据文件名
    #       already_cut - 是否已经进行了分词
    #       encoding - 数据文件编码
    def __init__(self, filename, already_cut=True, encoding='utf-8'):
        with open(filename, 'r', encoding=encoding) as f:
            for line in f.readlines():
                line = line.strip()
                if already_cut != True: #如果不是训练集，将句子直接保存到列表
                    self._data.append((line, None))
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

                self._data.append((''.join(sentence), ''.join(label)))

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

    def output_crfpp_format(self, filename=None):
        lines = []    
        for sent, label in self._data:
            for char, tag in zip(sent, label):
                line = '{}    {}'.format(char, tag)
                lines.append(line) 
            lines.append('')
    
        if filename is not None:
            with open(filename, 'w') as f:
                f.write('\n'.join(lines))
        else:
            print('\n'.join(lines))


#a = Dataset('pku_training.utf8')
#for sent in a.data():
#    print(sent)


a = Dataset('pku_training.utf8')
for i in range(5):
    print()
    #print(a[i])
    
#print(a[0:5])

a.output_crfpp_format('pku_crf_train.txt')
#a.output_crfpp_format()
