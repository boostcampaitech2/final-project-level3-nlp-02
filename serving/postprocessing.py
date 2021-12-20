import re
from kss import split_sentences

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

class TitlePostProcessor :
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

        # titles = split_sentences(title)
        # title = "".join(titles[:-1]) if len(titles) != 1 else titles[0]
        title = title.rstrip()
        title = self.special_char.sub('', title)
        return title

if __name__ == "__main__" :
    text = "도애 홍석모의 금강산 유기 - (간관록 일고"
    pcs = TitlePostProcessor()
    title = pcs.post_process(text)
    print(title)