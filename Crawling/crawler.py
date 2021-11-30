import os
import re
import time
import json
import copy
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

import requests
from bs4 import BeautifulSoup
from selenium import webdriver

def check_title(title, chn_range) :
    chars = re.sub('\s+' , '', title)
    sep_chars = [ch for ch in chars if ord(ch) in chn_range] 
    return True if len(sep_chars) >= 3 else False     

def get_original(title, driver) :
    element = driver.find_element_by_name('query')
    element.clear()

    time.sleep(np.random.uniform(0.0, 2.0))
    element.send_keys(title)
    element.submit()
    url = driver.current_url

    response = requests.get(url)
    bs = BeautifulSoup(response.content, 'html.parser')

    paper_info = bs.findAll('div', {'class' , 'ui_listing_info'})

    if len(paper_info) == 0 :
        return None
    else :
        paper_name = paper_info[0].find('a').text
        return paper_name

def main() :

    # -- Load Dataset
    paper_data = load_dataset('metamong1/summarization_paper', 
        download_mode='force_redownload',
        use_auth_token='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')

    chn_range = range(ord('一'), ord('鿕')+1)

    train_data = list(paper_data['train'])
    train_titles = {i:data['title'] for i, data in enumerate(train_data) \
        if check_title(data['title'], chn_range)}

    val_data = list(paper_data['validation'])
    val_titles = {i:data['title'] for i, data in enumerate(val_data) \
        if check_title(data['title'], chn_range)}

    chrome_path = '/Users/sanghapark/Desktop/Project/chromedriver'
    driver = webdriver.Chrome(chrome_path)
    driver.implicitly_wait(1.5)
    driver.get('https://academic.naver.com/')

    def change_name(index2title, dataset, driver) :
        num_chnaged = 0
        for idx, title in tqdm(index2title.items()) :
            try :
                org_name = get_original(title, driver)
                if org_name != None :
                    dataset[idx]['title'] = org_name
                    num_chnaged += 1
                else :
                    title = title[:10] # make narrow query
                    org_name = get_original(title, driver)
                    if org_name == None :
                        continue

                    dataset[idx]['title'] = org_name
                    num_chnaged += 1

            except :
                continue

        print('Changed Data : %d' %num_chnaged)
        return dataset

    print('\n Find orginal Name of train datasets')
    train_data = change_name(train_titles, train_data, driver)
    with open('../Data/train_data.json', 'w') as f :
        json.dump(train_data, f) 
    
    print('\nFind orginal Name of validation datasets')
    val_data = change_name(val_titles, val_data, driver)
    with open('../Data/val_data.json', 'w') as f :
        json.dump(val_data, f)

if __name__ == '__main__' :
    main()