import os
import re
import time
import json
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

import requests
from bs4 import BeautifulSoup

def check_title(title) :
    org_chars = re.sub('\s+', '' , title) 
    target_chars = re.findall('[a-zA-Zぁ-ゔァ-ヴー々〆〤一-龥]', org_chars)

    char_rate = len(target_chars) / len(org_chars)
    return True if char_rate >= 0.1 else False

  
def get_original(base_url, title) :
    target_rul = base_url + title

    time.sleep(np.random.uniform(0.0, 1.0))
    response = requests.get(target_rul)
    bs = BeautifulSoup(response.content, 'html.parser')

    div_list = bs.find_all('div', {'class' : 'ui_listing_info'})

    if len(div_list) == 0 :
        return None
    else :
        title = div_list[0].find('a', {'class' : 'ui_listing_subtit'}).text
        return title

def main() :

    # -- Load Dataset
    paper_data = load_dataset('metamong1/summarization_paper', 
        download_mode='force_redownload',
        use_auth_token='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')

    train_data = list(paper_data['train'])
    train_titles = {i:data['title'] for i, data in enumerate(train_data) \
        if check_title(data['title'])}

    val_data = list(paper_data['validation'])
    val_titles = {i:data['title'] for i, data in enumerate(val_data) \
        if check_title(data['title'])}

    base_url = 'https://academic.naver.com/search.naver?field=0&docType=1&query='
    def change_name(index2title, dataset) :
        num_changed = 0
        items = list(index2title.items())
        print('Size : %d' %len(items))
        pbar = tqdm(range(len(items)))

        for i in pbar :
            idx, title = items[i]
            try :
                org_name = get_original(base_url, title)
                if org_name != None :
                    dataset[idx]['title'] = org_name
                    num_changed += 1
            except :
                continue

            pbar.set_description("Changed : %d" %num_changed)
        
        return dataset

    print('\n Find orginal Name of train datasets')
    train_data = change_name(train_titles, train_data)
    with open('../Data/train_data.json', 'w') as f :
        json.dump(train_data, f) 
    
    print('\nFind orginal Name of validation datasets')
    val_data = change_name(val_titles, val_data)
    with open('../Data/val_data.json', 'w') as f :
        json.dump(val_data, f)

if __name__ == '__main__' :
    main()