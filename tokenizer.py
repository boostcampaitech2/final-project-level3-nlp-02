
import os
import json
import pandas as pd
from transformers import AutoTokenizer


class ToknierOptimization :
    def __init__(self, dir_path) :
        self.dir_path = dir_path

    def load_txt(self, path) : 
        with open(path, "r") as f :
            data = f.read()
        return data

    def write_txt(self, vocab_list, path) :
        data_size = len(vocab_list)
        f = open(path, 'w')
        for i in range(data_size):
            f.write(vocab_list[i]+'\n')
        f.close()

    def load_json(self, path) :
        with open(path, "r") as f:
            data = json.load(f)
        return data

    def write_json(self, tokenizer_data, path) :
        with open(path, 'w') as f:
            json.dump(tokenizer_data, f)


class BertTokenizerOptimization(ToknierOptimization) :
    def __init__(self, dir_path) :
        super().__init__(dir_path)
        self.tokenizer_path = os.path.join(dir_path, 'tokenizer.json')
        self.vocab_path = os.path.join(dir_path, 'vocab.txt')

    # update vocab mapping
    def update_unused(self, idx2vocab, ch_df) :
        unused_start = 31500
        unused_size = 300
        
        size = 0
        map_id = 0
        while(size < unused_size) :
            ch, flag = ch_df.iloc[map_id][['Character', 'KoreanFlag']]

            target_id = unused_start + size
            if flag == False :
                idx2vocab[target_id] = ch
                size += 1
            else :
                if size + 2 >= unused_size :
                    map_id += 1
                    continue

                idx2vocab[target_id] = ch
                idx2vocab[target_id+1] = '##' + ch
                size += 2
                
            map_id += 1

        return idx2vocab

    # optimizing tokenizer
    def optimize(self, extra_ch_path) :
        ch_df = pd.read_csv(extra_ch_path)

        # load tokenizer data
        tokenizer_data = self.load_json(self.tokenizer_path)
        vocab_data = self.load_txt(self.vocab_path)
        idx2vocab = {i:vocab for i, vocab in enumerate(vocab_data.split('\n')[:-1])}
        
        # update tokenizer
        idx2vocab = self.update_unused(idx2vocab, ch_df)
        vocab_list = list(idx2vocab.values())
        vocab2idx = {idx2vocab[key] : key for key in idx2vocab.keys()}
        tokenizer_data['model']['vocab'] = vocab2idx

        # write tokenizer data
        self.write_txt(vocab_list, self.vocab_path)
        self.write_json(tokenizer_data, self.tokenizer_path)

        # load updated tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_path, use_fast=True)
        return tokenizer


class BartTokenizerOptimization(ToknierOptimization) :
    def __init__(self, dir_path) :
        super().__init__(dir_path)
        self.tokenizer_path = os.path.join(dir_path, 'tokenizer.json')
        self.vocab_path = os.path.join(dir_path, 'vocab.json')

    # update vocab mapping
    def update_unused(self, idx2vocab, ch_df) :
        unused_start = 7
        unused_size = 100
        
        size = 0
        map_id = 0
        while(size < unused_size) :
            ch, flag = ch_df.iloc[map_id][['Character', 'KoreanFlag']]

            target_id = unused_start + size
            if flag == False :
                idx2vocab[target_id] = ch
                size += 1
            else :
                if size + 2 >= unused_size :
                    map_id += 1
                    continue

                idx2vocab[target_id] = ch
                idx2vocab[target_id+1] = '▁' + ch
                size += 2
                
            map_id += 1

        return idx2vocab

    def update_added_tokens(self, tokenizer_data, vocab_list) :
        unused_start = 7
        unused_size = 100
        for i in range(unused_start, unused_start + unused_size) :
            tokenizer_data['added_tokens'][i]['content'] = vocab_list[i]
        return tokenizer_data

    # optimizing tokenizer
    def optimize(self, extra_ch_path) :
        ch_df = pd.read_csv(extra_ch_path)

        # load tokenizer data
        tokenizer_data = self.load_json(self.tokenizer_path)
        vocab2idx = self.load_json(self.vocab_path)
        idx2vocab = {idx:vocab for vocab,idx in vocab2idx.items()}
        # update tokenizer
        idx2vocab = self.update_unused(idx2vocab, ch_df)
        vocab2idx = {idx2vocab[key] : key for key in idx2vocab.keys()}
        vocab_list = list(idx2vocab.values())
        
        tokenizer_data['model']['vocab'] = vocab2idx
        tokenizer_data = self.update_added_tokens(tokenizer_data, vocab_list)

        # write tokenizer data
        self.write_json(vocab2idx, self.vocab_path)
        self.write_json(tokenizer_data, self.tokenizer_path)

        # load updated tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_path, use_fast=True)
        return tokenizer



