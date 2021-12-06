
import re
from abc import *

class Preprocessor(metaclass=ABCMeta) :
    def __init__(self ) :
        # 일본어, 한국어, 한자, 기본 문자, 구두점, 문장 기호
        self.private_comp = re.compile('[^\ue000-\uf8ff]')
        self.outrange_comp = re.compile('[^\u3040-\u30ff\
            \uac00-\ud7af\
            \u4e00-\u9fff\
            \u0000-\u007f\
            \u2000-\u206f\
            \u25a0-\u25ff]') 

    @abstractmethod
    def for_train(self, data) :
        pass

    @abstractmethod
    def for_test(self, data) :
        pass

    def strip(self, txt) :
        txt = re.sub('\s+' , ' ', txt) 
        return txt.strip()

    def check_data(self, data) :
        if 'text' not in data.keys() or 'title' not in data.keys() :
            raise KeyError('Wrong Data keys')

    def doc_preprocess(self, txt) :
        txt = self.private_comp.sub(' ', txt)
        txt = self.outrange_comp.sub(' ', txt)
        return txt

class DocsPreprocessor(Preprocessor) :
    def __init__(self) :
        super().__init__()
        self.bracket_comp = re.compile(r"\([^)]+\)")

    def for_train(self, data) :
        self.check_data(data)
        title = data['title']
        title = self.bracket_comp.sub(' ', title)
        title = self.doc_preprocess(title)
        title = self.strip(title)

        text = data['text']
        text = self.bracket_comp.sub(' ', text)
        text = self.doc_preprocess(text)
        text = self.strip(text)

        data['text'] = text 
        data['title'] = title
        return data

    def for_test(self, data) :
        self.check_data(data)
        text = data['text']
        text = self.bracket_comp.sub(' ', text)
        text = self.doc_preprocess(text)
        text = self.strip(text)
        data['text'] = text 
        return data


class Filter :
    def __init__(self, min_size, max_size) :
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, data) :
        self.check_data(data)
        if len(data['title']) < self.min_size or len(data['title']) > self.max_size:
            return False
        return True

    def check_data(self, data) :
        if 'text' not in data.keys() or 'title' not in data.keys() :
            raise KeyError('Wrong Data keys')
