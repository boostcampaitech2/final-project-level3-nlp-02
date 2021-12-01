
import os
import json
import pandas as pd
from transformers import PreTrainedTokenizerFast

class TokenizerOptimization :
    def __init__(self, dir_path, word_list) :
        self.txt_path = os.path.join(dir_path, 'vocab.txt')
        self.json_path = os.path.join(dir_path, 'tokenizer.json')
        self.vocab_map = {i: word for i, word in enumerate(word_list)}

    def write_vocab(self, vocab_map) :
        data_size = len(vocab_map)
        vocab_list = list(vocab_map.values())
        f = open(self.txt_path, 'w')
        for i in range(data_size):
            f.write(vocab_list[i]+'\n')
        f.close()

    def add_unused(self, vocab_map, tokenizer) :
        unused_start = tokenizer.convert_tokens_to_ids('[unused0]')
        unused_end = tokenizer.convert_tokens_to_ids('[unused99]') + 1

        unused_size = unused_end - unused_start 
        for i in range(unused_size) :
            unused_idx = unused_start + i
            word = vocab_map[i]
            vocab_map[unused_idx] = word

    def load_tokenizer_json(self) :
        with open(self.json_path, "r") as json_data:
            tokenizer_data = json.load(json_data)
        return tokenizer_data

    def write_tokenizer_json(self, tokenizer_data, vocab_data) :
        inverse_vocab_data = {vocab_data[key] : key for key in vocab_data.keys()}
        tokenizer_data['model']['vocab'] = inverse_vocab_data
        with open(self.json_path, 'w') as json_file:
            json.dump(tokenizer_data, json_file)

    def optimize(self, tokenizer) :
        assert isinstance(tokenizer, PreTrainedTokenizerFast)
        self.add_unused(self.vocab_map, tokenizer)
        self.write_vocab(self.vocab_map)

        tokenizer_data = self.load_tokenizer_json()
        self.write_tokenizer_json(tokenizer_data, self.vocab_map)