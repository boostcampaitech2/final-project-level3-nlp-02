from konlpy.tag import Mecab
from MeCab import Tagger
from konlpy import utils

class CustomMecab(Mecab):
    def __init__(self, dicpath='/usr/local/lib/mecab/dic/mecab-ko-dic'):
        super(CustomMecab).__init__()
        self.dicpath = dicpath
        try:
            self.tagger = Tagger('-d %s' % dicpath)
            self.tagset = utils.read_json('%s/data/tagset/mecab.json' % utils.installpath)
        except RuntimeError:
            raise Exception('The MeCab dictionary does not exist at "%s". Is the dictionary correctly installed?\nYou can also try entering the dictionary path when initializing the Mecab class: "Mecab(\'/some/dic/path\')"' % dicpath)
        except NameError:
            raise Exception('Install MeCab in order to use it: http://konlpy.org/en/latest/install/')
    def usable_pos(self, phrase):
        """Noun extractor."""
        usable_tag = ('N','SN','SL') # 명사, 숫자, 외국어
        tagged = self.pos(phrase)
        return [s for s, t in tagged if t.startswith(usable_tag)]