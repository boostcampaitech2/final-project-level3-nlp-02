import re

def pair_check(
        text: str
    ) -> str:
    pair_dict = {
        "[":"]", "{":"}", "(":")","『":"』",
        "“":"”", "‘":"’",'\"':'\"', "\'":"\'"}
    pair_same = ['\"', "\'"]
    stack = []
    for i in text :
        if i in pair_dict.keys() :
            if (len(stack) > 0) and (stack[-1] in pair_same):
                stack.pop()
            else :
                stack.append(i)
        elif i in pair_dict.values():
            if (len(stack) > 0) and (i == pair_dict[stack[-1]]):
                stack.pop()
            elif len(stack) == 0 :
                return i
            else :
                return stack[-1]
    if len(stack) == 0:
        pass
    else :
        return stack[-1]

class PreProcessor :
    def __init__(self) :
        # 일본어, 한국어, 한자, 기본 문자, 구두점, 문장 기호
        self.private_comp = re.compile('[\ue000-\uf8ff]')
        self.outrange_comp = re.compile('[^\u3040-\u30ff\
            \uac00-\ud7af\
            \u4e00-\u9fff\
            \u0000-\u007f\
            \u2000-\u206f\
            \u25a0-\u25ff]') 

        self.bracket_comp = re.compile(r"\([^)]+\)")

    def strip(self, txt) :
        txt = re.sub('\s+' , ' ', txt) 
        return txt.strip()

    def doc_preprocess(self, txt) :
        txt = self.private_comp.sub(' ', txt)
        txt = self.outrange_comp.sub(' ', txt)
        return txt
    
    def pre_process(self,
        txt:str
    ) -> str :
        txt = self.bracket_comp.sub(' ', txt)
        txt = self.doc_preprocess(txt)
        txt = self.strip(txt)
        return txt

class PostProcessor :
    def __init__(self) :
        self.escaped_space = re.compile(r'\\r|\\n|\\\r|\\\n')
        self.special_char = re.compile(r' -|·$| /')
    
    def post_process(self, 
            title: str
        ) -> str:
        title = self.escaped_space.sub('', title)
        init_title_len = len(title)
        
        while True :
            unmatched_char = pair_check(title)
            if unmatched_char is None :
                break
            else :
                unmatched_idx = title.rfind(unmatched_char)
                if init_title_len // 2 < len(title[:unmatched_idx]) :
                    title = title[:unmatched_idx]
                else :
                    title = title[:unmatched_idx] + title[unmatched_idx+1:]

        title = title.rstrip()
        title = self.special_char.sub('', title)
        return title

if __name__ == "__main__" :
    text = "도애 홍석모의 금강산 유기, 『간관록』 일고 (陶厓 洪錫謨의 금강산 유기, 『艮觀錄』 一考)"
    pcs = PreProcessor()
    title = pcs.pre_process(text)

    # text = "도애 홍석모의 금강산 유기 - (간관록 일고"
    # pcs = PostProcessor()
    # title = pcs.post_process(text)
    print(title)