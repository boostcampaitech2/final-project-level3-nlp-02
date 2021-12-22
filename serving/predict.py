import sys
sys.path.append('..')

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PreTrainedTokenizer
from model.models.modeling_kobigbird_bart import EncoderDecoderModel
from model.models.modeling_longformer_bart import LongformerBartWithDoctypeForConditionalGeneration

import datasets
import numpy as np
from typing import List, Optional

doc_type_dict = {
    "논문" : 1,
    "기사" : 2,
    "잡지" : 3,
}

@st.cache(allow_output_mutation=True)
def load(checkpoint, model_name) :
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
    if "longformerbart" in model_name:
        model = LongformerBartWithDoctypeForConditionalGeneration.from_pretrained(checkpoint)
    elif "bigbart" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = EncoderDecoderModel.from_pretrained(checkpoint, output_attentions=True)
        model.encoder.config.output_attentions = True
        model.decoder.config.output_attentions = True
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint,
        output_attentions=True,
    )
    return tokenizer, model

def get_prediction(
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    doc_type: str,
    input_text: str,
    num_beam: int,
    data_args,
    gen_args
    ) -> str:
    with st.spinner('Wait for it...'):
        with torch.no_grad():
            processed_text = preprocess_function_for_prediction(input_text, doc_type, tokenizer, data_args)
            input_ids = {k: torch.tensor(v) for k,v in processed_text.items()}
            input_ids['input_ids'] = input_ids['input_ids'].unsqueeze(0)

            generated_tokens = model.generate(
                **input_ids, num_beams=num_beam)#, **gen_args.__dict__)
            return generated_tokens

def preprocess_function_for_prediction(text:str,
            doc_type:str,
            tokenizer:PreTrainedTokenizer,
            data_args) -> datasets:
                        
    pad_token_id = tokenizer.pad_token_id
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    max_source_length = data_args.max_source_length
    padding = "max_length" if data_args.pad_to_max_length else False
    
    # Setup the tokenizer for inputs
    model_input = tokenizer(text, max_length=max_source_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    inputs_padding_bool = (padding == "max_length")
    doc_type_ids = []
    model_input["attention_mask"] = add_padding(sample_tokens=model_input["attention_mask"],
                                                    padding=inputs_padding_bool,
                                                    padding_num=0,
                                                    max_length=max_source_length,
                                                    bos_token_id=1,
                                                    eos_token_id=1) 
    model_input["input_ids"] = add_padding(sample_tokens=model_input["input_ids"],
                                                    padding=inputs_padding_bool,
                                                    padding_num= pad_token_id,
                                                    max_length=max_source_length,
                                                    bos_token_id = bos_token_id,
                                                    eos_token_id = eos_token_id)
    if data_args.use_doc_type_ids:
        doc_type_id_list = get_doc_type_ids(model_input["attention_mask"], doc_type_dict[doc_type])
        doc_type_ids.append(doc_type_id_list)
    
    if data_args.use_doc_type_ids:
        model_input["doc_type_ids"] = doc_type_ids

    del model_input["attention_mask"]
    del model_input["token_type_ids"]

    return model_input

def get_doc_type_ids(sample_tokens:List[int],
                     doc_type_id:int) -> List:
    sample_tokens = np.array(sample_tokens)
    doc_type_id_list = list(np.where(sample_tokens == 1, doc_type_id, 0))
    return doc_type_id_list
    
def add_padding(sample_tokens:List[int],
                padding:bool,
                padding_num:int,
                max_length:Optional[int],
                bos_token_id:int,
                eos_token_id:int) -> List:
    sample_tokens_len = len(sample_tokens)
    if len(sample_tokens) > max_length - 2:
        if bos_token_id == 0: #bart tokenizer만 진행
            sample_tokens = [bos_token_id] + sample_tokens[:max_length-2] + [eos_token_id]
    else:
        if bos_token_id == 0: #bart tokenizer만 진행
            sample_tokens = [bos_token_id] + sample_tokens + [eos_token_id] # + [padding_num]*(max_length-sample_tokens_len-2)
        if padding:
            sample_tokens = sample_tokens + [padding_num]*(max_length-sample_tokens_len-2)
    return sample_tokens