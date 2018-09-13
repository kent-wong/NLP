import xml.sax
import logging
from pprint import pprint
from gensim import corpora, models, similarities
import jieba
import codecs
import os
import pickle

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#all_titles = []
#all_files = []

class NewsHandler(xml.sax.ContentHandler):
    def __init__(self, all_files, all_titles):
        self.all_files = all_files
        self.all_titles = all_titles
        self.cur_tag = ""
        self.title = ""

    def startElement(self, tag, attributes):
        self.cur_tag = tag
        #print("<" + tag + ">")
    
    def endElement(self, tag):
        if tag == "content":
            self.cur_tag = ""
            self.title = ""
        #print("</" + tag + ">")

    def characters(self, txt):
        txt = txt.strip()
        if len(txt) == 0: # 跳过内容为空的文章
            return

        # 只需要文章标题和文章内容
        if self.cur_tag == "contenttitle":
            self.title = txt
        elif self.cur_tag == "content":
            if self.title == "":
                return

            # 标题和内容必须同时添加，保持对应列表长度一致
            self.all_titles.append(self.title)
            self.all_files.append(txt)

            
class NewsCorpus(object):
    def __init__(self, filename=None, stopwords=None):
        self.dictionary = None
        self.corpus = None
        self.all_files = []
        self.all_titles = []

        if filename != None:
            self.parseXML(filename)
            stoplist = None
            if stopwords != None:
                f = codecs.open(stopwords, 'r', encoding='utf8')
                stoplist = f.read().split()
                f.close()

            self.bowCorpus(stoplist)

    def __iter__(self):
        for text in self.corpus:
            yield text

    def __len__(self):
        print("__len__() call!", len(self.corpus))
        return len(self.corpus)

    def __getitem__(self, key):
        print("!!! __getitem__() called!!!, key=", key)

    def parseXML(self, xmlfile, encoding='utf8'):
        f = codecs.open(xmlfile, 'r', encoding)
        xml.sax.parseString(f.read().replace('&', '&amp;'), NewsHandler(self.all_files, self.all_titles))

    def bowCorpus(self, stoplist=None):
        for i in range(len(self.all_files)):
            afile = list(jieba.cut(self.all_files[i]))
            filtered = [x for x in afile if x not in stoplist]
            self.all_files[i] = filtered

        self.dictionary = corpora.Dictionary(self.all_files)
        self.corpus = [self.dictionary.doc2bow(text) for text in self.all_files]

    def save(self):
        if self.dictionary != None and self.corpus != None:
            self.dictionary.save('my_corpus.dict')
            corpora.MmCorpus.serialize('my_corpus.mm', self.corpus)
            f = open('my_corpus.titles', 'wb')
            pickle.dump(self.all_titles, f)
            f.close()

    def load(self):
        dict_file = 'my_corpus.dict'
        mm_file = 'my_corpus.mm'
        title_file = 'my_corpus.titles'
        if os.path.exists(dict_file) and os.path.exists(mm_file) and os.path.exists(title_file):
            self.dictionary = corpora.Dictionary.load(dict_file)
            self.corpus = corpora.MmCorpus(mm_file)
            f = open(title_file, 'rb')
            self.all_titles = pickle.load(f)
            f.close()
        else:
            print('**error**: dict or corpus file NOT found!')


#mycorpus = NewsCorpus("news_tensite_xml.dat", "stopwords.txt")
#mycorpus.save()
mycorpus = NewsCorpus()
mycorpus.load()

#print(mycorpus.dictionary)
#print(mycorpus.corpus)
#print(mycorpus.all_titles)
print("all titles len:", len(mycorpus.all_titles))

print("!!parse() ok.")

#for id, afile in enumerate(mycorpus):
    #print(id, afile)

print("length of mycorpus:", len(mycorpus))

tfidf = models.TfidfModel(mycorpus)
print(tfidf)
tfidf_corpus = tfidf[mycorpus]

print("begin lda training")

lda = models.LdaModel(tfidf_corpus, id2word=mycorpus.dictionary, num_topics=20)
#lda = models.LdaModel(tfidf_corpus, num_topics=10)
print()
print('-------------------------------------')
lda.print_topics()
print()
print('======================================')
lda_corpus = lda[tfidf_corpus]
corpora.MmCorpus.serialize('lda_corpus.mm', lda_corpus)
#for topic in topic_test:
    #print(topic)
