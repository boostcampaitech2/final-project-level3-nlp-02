
import os
import json
import pandas as pd
from transformers import AutoTokenizer

class TokenizerOptimization :
    def __init__(self, dir_path) :
        self.dir_path = dir_path
        self.tokenizer_path = os.path.join(dir_path, 'tokenizer.json')
        self.vocab_path = os.path.join(dir_path, 'vocab.txt')
        
    # load tokenizer data
    def load_tokenizer(self,path) :
        with open(path, "r") as f:
            data = json.load(f)
        return data

    # load vocab data
    def load_vocab(self, path) : 
        with open(path, "r") as f :
            data = f.read()
        idx2vocab = {i:vocab for i, vocab in enumerate(data.split('\n')[:-1])}
        return idx2vocab

    # update vocab mapping
    def update_unused(self, idx2vocab, ch_df) :
        unused_start = 31500
        
        size = 0
        map_id = 0
        while(size < 500) :
            ch, flag = ch_df.iloc[map_id][['Character', 'KoreanFlag']]

            target_id = unused_start + size
            if flag == False :
                idx2vocab[target_id] = ch
                size += 1
            else :
                if size + 2 >= 500 :
                    map_id += 1
                    continue

                idx2vocab[target_id] = ch
                idx2vocab[target_id+1] = '##' + ch
                size += 2
                
            map_id += 1

        return idx2vocab

    # writing updated vocab
    def update_vocab(self, idx2vocab, path) :
        data_size = len(idx2vocab)
        vocab_list = list(idx2vocab.values())
        f = open(path, 'w')
        for i in range(data_size):
            f.write(vocab_list[i]+'\n')
        f.close()

    # writing updated tokenizer
    def update_tokenizer(self, idx2vocab, tokenizer_data, path) :
        vocab2idx = {idx2vocab[key] : key for key in idx2vocab.keys()}
        tokenizer_data['model']['vocab'] = vocab2idx
        with open(path, 'w') as f:
            json.dump(tokenizer_data, f)


    def optimize(self, extra_ch_path) :
        ch_df = pd.read_csv(extra_ch_path)

        tokenizer_data = self.load_tokenizer(self.tokenizer_path)
        idx2vocab = self.load_vocab(self.vocab_path)

        idx2vocab = self.update_unused(idx2vocab, ch_df)
        self.update_vocab(idx2vocab, self.vocab_path)
        self.update_tokenizer(idx2vocab, tokenizer_data, self.tokenizer_path)

        tokenizer = AutoTokenizer.from_pretrained(self.dir_path, use_fast=True)
        return tokenizer

