
import re
from abc import *

class Preprocessor(metaclass=ABCMeta) :
    def __init__(self ) :
        # 일본어, 한국어, 한자, 기본 문자, 구두점, 문장 기호
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
        txt = self.outrange_comp.sub(' ', txt)
        return txt

class PaperPreprocessor(Preprocessor) :
    def __init__(self) :
        super().__init__()
        self.bracket_comp = re.compile(r"\([^)]+\)")

    def for_train(self, data) :
        self.check_data(data)
        title = data['title']
        title = self.add_bracket(title)
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

    def add_bracket(self, title) :
        if '(' in title and ')' not in title :
            return title + ')'
        else :
            return title   

class DocsPreprocessor(Preprocessor) :
    def __init__(self) :
        super().__init__()

    def for_train(self, data) :
        self.check_data(data)
        title = data['title'] # title preprocessing
        title = self.doc_preprocess(title)
        title = self.strip(title)

        text = data['text'] # text preprocessing
        text = self.doc_preprocess(text)
        text = self.strip(text)

        data['text'] = text 
        data['title'] = title
        return data

    def for_test(self, data) :
        self.check_data(data)
        text = data['text']
        text = self.doc_preprocess(text)
        text = self.strip(text)
        data['text'] = text 
        return data


class Filter :
    def __init__(self, title_size) :
        self.title_size = title_size
        self.kor_comp = re.compile('[가-힣]')

    def __call__(self, data) :
        self.check_data(data)
        if len(data['title']) < self.title_size :
            return False
            
        kor_rate = self.get_kor_rate(data)
        return True if kor_rate >= 0.5 else False

    def get_kor_rate(self, data) :
        title = data['title']
        title = re.sub('\s+' , '', title)
        kor_chars = self.kor_comp.findall(title)
        kor_rate = len(kor_chars) / len(title)
        return kor_rate

    def check_data(self, data) :
        if 'text' not in data.keys() or 'title' not in data.keys() :
            raise KeyError('Wrong Data keys')
