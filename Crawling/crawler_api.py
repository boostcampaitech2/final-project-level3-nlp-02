
import os
import sys
import re
import json
import urllib.request
from tqdm import tqdm
from datasets import load_dataset


def check_title(title, comp) :
    org_chars = re.sub('\s+', '' , title) 
    target_chars = comp.findall(org_chars)

    char_rate = len(target_chars) / len(org_chars)
    return True if char_rate >= 0.3 else False

def extract_title(content) :
    content = json.loads(content)
    title = content['items'][0]['title']

    title = re.sub('<b>', ' ', title)
    title = re.sub('</b>', ' ', title)
    title = re.sub('\s+' , ' ', title).strip()
    return title

def main() :
    client_id = "FcfC2Ls3KNm5z_Zd9OG_"
    client_secret = "uV2UxplHq9"

    # -- Load Dataset
    paper_data = load_dataset('metamong1/summarization_paper', 
        use_auth_token='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')

    target_comp = re.compile('[a-zA-Zぁ-ゔァ-ヴー々〆〤一-龥]')

    train_data = list(paper_data['train'])
    train_titles = {i:data['title'] for i, data in enumerate(train_data) \
        if check_title(data['title'], target_comp)}

    val_data = list(paper_data['validation'])
    val_titles = {i:data['title'] for i, data in enumerate(val_data) \
        if check_title(data['title'], target_comp)}

    def get_name(index2title, dataset) :
        num_changed = 0
        for idx, title in tqdm(index2title.items()) :
            try :
                encText = urllib.parse.quote(title)
                url = "https://openapi.naver.com/v1/search/doc?query=" + encText # json 결과
                request = urllib.request.Request(url)
                request.add_header("X-Naver-Client-Id",client_id)
                request.add_header("X-Naver-Client-Secret",client_secret)
                response = urllib.request.urlopen(request)
                rescode = response.getcode()

                if (rescode == 200) :
                    response_body = response.read()
                    response_content = response_body.decode('utf-8')

                    org_name = extract_title(response_content)
                    if org_name != 'Title Page' :
                        dataset[idx]['title'] = org_name
                        num_changed += 1   
                else :
                    continue
            except :
                continue
        print('Changed Data : %d' %num_changed)
        return dataset

    print('\n Find orginal Name of train datasets')
    train_data = get_name(train_titles, train_data)
    with open('../Data/train_data.json', 'w') as f :
        json.dump(train_data, f) 
    
    print('\nFind orginal Name of validation datasets')
    val_data = get_name(val_titles, val_data)
    with open('../Data/val_data.json', 'w') as f :
        json.dump(val_data, f)

if __name__ == '__main__' :
    main()