
import re
from abc import *
from datasets.arrow_dataset import Example

class Preprocessor(metaclass=ABCMeta) :
    def __init__(self ) :
        # 일본어, 한국어, 한자, 기본 문자, 구두점, 문장 기호
        self.outrange_comp = re.compile('[^\u3040-\u30ff\
            \uac00-\ud7af\
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
        assert isinstance(data, Example)
        if 'text' not in data.keys() or 'title' not in data.keys() :
            raise KeyError('Wrong Data keys')

    def doc_preprocess(self, txt) :
        txt = self.outrange_comp.sub(' ', txt)
        return txt

class PaperPreprocessor(Preprocessor) :
    def __init__(self) :
        super().__init__()
        self.bracket_comp = re.compile(r"\([^)]+\)")
        self.kor_range = range(ord('가'), ord('힣')+1)

    def for_train(self, data) :
        self.check_data(data)
        title = data['title'] # title preprocessing
        title = self.add_bracket(title)
        title = self.bracket_comp.sub(' ', title)
        title = self.remove_descript(title)
        title = self.doc_preprocess(title)
        title = self.strip(title)

        text = data['text'] # text preprocessing
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

    def remove_descript(self, title) :
        for i in range(len(title)-1, 0, -1) :
            ch = title[i]
            if ord(ch) in self.kor_range :
                break
        return title[:i+1]

class DocsPreprocessor(Preprocessor) :
    def __init__(self) :
        super().__init__()

    def base_preprocess(self, text) :
        text = re.sub('\"\"', ' ', text)
        text = re.sub(r'\n', ' ', text)
        text = re.sub('[▶▲△]', ', ', text)
        return text

    def for_train(self, data) :
        self.check_data(data)
        title = data['title'] # title preprocessing
        title = self.base_preprocess(title)
        title = self.doc_preprocess(title)
        title = self.strip(title)

        text = data['text'] # text preprocessing
        text = self.base_preprocess(text)
        text = self.doc_preprocess(text)
        text = self.strip(text)

        data['text'] = text 
        data['title'] = title
        return data

    def for_test(self, data) :
        self.check_data(data)
        text = data['text']
        text = self.base_preprocess(text)
        text = self.doc_preprocess(text)
        text = self.strip(text)
        data['text'] = text 
        return data

class Filter :
    def __init__(self, max_text_size, min_title_size) :
        self.max_text_size = max_text_size
        self.min_title_size = min_title_size
        self.kor_comp = re.compile('[가-힣]')

    def __call__(self, data) :
        self.check_data(data)
        return self.check_size(data) and self.check_title(data)

    def check_size(self, data) :
        if len(data['text']) <= self.max_text_size and len(data['title']) >= self.min_title_size :   
            return True
        else :
            return False 

    def check_title(self, data) :
        title = data['title']
        title = re.sub('\s+' , '', title)
        kor_chars = self.kor_comp.findall(title)
        kor_rate = len(kor_chars) / len(title)
        return True if kor_rate >= 0.5 else False

    def check_data(self, data) :
        assert isinstance(data, dict)
        if 'text' not in data.keys() or 'title' not in data.keys() :
            raise KeyError('Wrong Data keys')
      

