import re
from kss import split_sentences

def pair_check(text) -> str:
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
    def __init__(self, title) :
        self.escaped_space = re.compile(r'\\r|\\n|\\\r|\\\n')
        self.title = title
    
    def post_process(self) :
        title = self.escaped_space.sub('', self.title)
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

        titles = split_sentences(title)
        title = "".join(titles[:-1]) if len(titles) != 1 else titles[0]
        title = title.rstrip()
        title = title[:-2] if title[-2:] == ' -' else title 
        return title

if __name__ == "__main__" :
    text = "홍역 전국 확산...당진지역 감염 주의보!\\r\\n(홍역"
    pcs = TitlePostProcessor(text)
    title = pcs.post_process()
    print(title)
