import xml.sax
import logging
from pprint import pprint
from gensim import corpora, models, similarities
import jieba
import codecs

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

all_titles = []
all_files = []

class NewsHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.cur_tag = ""
        self.title = ""
        #self.content = ""

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
            assert(self.title != "")
            # 标题和内容必须同时添加，保持对应列表长度一致
            all_titles.append(self.title)
            all_files.append(txt)

            
class NewsCorpus(object):
    def __init__(self, filename, stopwords=None):
        self.filename = filename
        if stopwords != None:
            f = codecs.open(stopwords, 'r', encoding='utf8')
            self.stopwords = f.read().split()
            f.close()
        else:
            self.stopwords = []

    def clean(self):
        pass

    def __iter__(self):
        for content in all_files:
            yield content

    def parse(self):
        #parser = xml.sax.make_parser()

        # turn off namespaces
        #parser.setFeature(xml.sax.handler.feature_namespaces, 0)

        #Handler = NewsHandler()
        #parser.setContentHandler(Handler)
        #print(self.file.read())

        #source = xml.sax.InputSource()
        #source.setSystemId(self.filename)
        #source.setEncoding("utf-8")
        #source.setCharacterStream(self.file)
        #parser.parse(source)

        f = codecs.open(self.filename, 'r', encoding='utf8')
        Handler = NewsHandler()
        xml.sax.parseString(f.read().replace('&', '&amp;'), Handler)

a = NewsCorpus("news_tensite_xml.smarty.txt", "stopwords.txt")
a.parse()
for content in a:
    print()
    print(content)
