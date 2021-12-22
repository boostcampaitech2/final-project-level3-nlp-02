import torch
import math
from typing import List, Tuple, Optional

def split_tensor_by_words(
        text_tokens: List[str],
        model_type: str
    ) -> List[int] :
    i = 0
    split_words_indices = []
    if model_type != 'kobigbirdbart' :    
        for token in text_tokens :
            if '▁' in token:
                split_words_indices.append(i)
                i = 1
            else : 
                i += 1
        split_words_indices.append(i)
    else :
        cnt = 0
        for token in text_tokens :
            cnt += 1
            if '##' in token :
                i += 1
            else :
                split_words_indices.append(i)
                i = 1
        split_words_indices.append(i)
    split_words_indices = split_words_indices[1:]
    return split_words_indices

def token_to_words(
        text_tokens: List[str],
        model_type: str
    ) -> List[str] :
    if model_type != 'kobigbirdbart' :   
        join_text = ''.join(text_tokens).replace('▁', ' ')    
        space_text = join_text.split(' ')[1:]
    else :
        join_text = ' '.join(text_tokens).replace(' ##', '')
        space_text = join_text.split(' ')
    return space_text

def format_attention(
        attention,
        layers: Optional[int]=None,
        heads: Optional[int]=None
    ) -> torch.Tensor : # (layer, head, dec, enc)
    if layers:
        attention = [attention[layer_index] for layer_index in layers]
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                             "output_attentions=True when initializing your model.")
        layer_attention = layer_attention.squeeze(0)
        if heads:
            layer_attention = layer_attention[heads]    
        squeezed.append(layer_attention)
    return torch.stack(squeezed)

def model_forward(model, tokenizer, text, title) :
    enc_input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    dec_input_ids = tokenizer(title, return_tensors="pt", add_special_tokens=False).input_ids

    outputs = model(input_ids=enc_input_ids, decoder_input_ids=dec_input_ids)
    
    st_cross_attn = format_attention(outputs.cross_attentions)
    return st_cross_attn, enc_input_ids, dec_input_ids

def position(node_num):
    rad = math.radians(360/node_num)
    x_pos = [math.cos(rad*node)*1000+500 for node in range(node_num)]
    y_pos = [math.sin(rad*node)*1000+500 for node in range(node_num)]
    return x_pos, y_pos