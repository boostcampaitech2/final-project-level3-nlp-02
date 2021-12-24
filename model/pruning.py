import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoConfig
# from optimization import performance_test

def pruning(checkpoint,save_path):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint,use_auth_token=True)
    config = AutoConfig.from_pretrained(checkpoint)
    model=AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    model.encoder.encoder.layer = nn.Sequential(*list(model.encoder.encoder.layer.children())[:-3])
    model.decoder.model.decoder.layers = nn.Sequential(*list(model.decoder.model.decoder.layers.children())[:-3])
    
    config.encoder.num_hidden_layers = 3
    config.decoder.decoder_layers = 3

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    config.save_pretrained(save_path)

if __name__=="__main__":
    checkpoint = input("checkpoint:")
    save_path = input("save_path:")
    pruning(checkpoint,save_path)