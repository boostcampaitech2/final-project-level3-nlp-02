import torch
import os
import torch.nn as nn
from torch.quantization import quantize_dynamic
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoConfig
from dotenv import load_dotenv
# from optimization import performance_test

USE_AUTH_TOKEN = os.getenv("USE_AUTH_TOKEN")

def pruning(checkpoint,save_path):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_auth_token=USE_AUTH_TOKEN) 
    config = AutoConfig.from_pretrained(checkpoint, use_auth_token=USE_AUTH_TOKEN)
    model=AutoModelForSeq2SeqLM.from_pretrained(checkpoint, use_auth_token=USE_AUTH_TOKEN)

    model.encoder.encoder.layer = nn.Sequential(*list(model.encoder.encoder.layer.children())[:-3])
    model.decoder.model.decoder.layers = nn.Sequential(*list(model.decoder.model.decoder.layers.children())[:-3])
    
    config.encoder.num_hidden_layers = 3
    config.decoder.decoder_layers = 3

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    config.save_pretrained(save_path)

if __name__=="__main__":
    checkpoint = 'metamong1/bigbart_tapt_ep3_bs16_pre_noam' # input("checkpoint:")
    save_path = 'checkpoint/encoder_decoder_pruned_last_3' # input("save_path:")
    pruning(checkpoint,save_path)