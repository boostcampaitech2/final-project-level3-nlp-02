import numpy as np
from typing import List, Optional
from transformers import PreTrainedTokenizer
import datasets

doc_type_dict = {
    "논문" : 1,
    "신문기사" : 2,
    "사설잡지" : 3,
}

def preprocess_function(examples:datasets,
                        tokenizer:PreTrainedTokenizer,
                        data_args) -> datasets:
                        
    pad_token_id = tokenizer.pad_token_id
    max_source_length = data_args.max_source_length
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False
    inputs = examples['text']
    titles = examples['title']
    if data_args.use_doc_type_ids:
        doc_types = examples['doc_type'] 
    
    # Setup the tokenizer for inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    inputs_padding_bool = (padding == "max_length")
    doc_type_ids = []
    for i in range(len(model_inputs['input_ids'])) :
        if inputs_padding_bool:
            model_inputs["attention_mask"][i] = add_padding(sample_tokens=model_inputs["attention_mask"][i],
                                                            padding=inputs_padding_bool,
                                                            padding_num= pad_token_id,
                                                            max_length=max_target_length) 
            model_inputs["input_ids"][i] = add_padding(sample_tokens=model_inputs["input_ids"][i],
                                                            padding=inputs_padding_bool,
                                                            padding_num= pad_token_id,
                                                            max_length=max_target_length)
        
        if data_args.use_doc_type_ids:
            doc_type_id_list = get_doc_type_ids(model_inputs["attention_mask"][i], doc_type_dict[doc_types[i]])
            doc_type_ids.append(doc_type_id_list)

    if not data_args.is_pretrain:
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(titles, max_length=max_target_length-1, padding=padding, truncation=True)
        title_padding_bool = padding == "max_length" and data_args.ignore_pad_token_for_loss

        if title_padding_bool:
            labels["input_ids"] = [
                    add_padding(
                        sample_tokens=label,
                        padding=title_padding_bool,
                        padding_num=-100,
                        max_length=max_target_length,
                        ) for label in labels["input_ids"]
                ]
        model_inputs["labels"] = labels["input_ids"]
    if data_args.use_doc_type_ids:
        model_inputs["doc_type_ids"] = doc_type_ids
    return model_inputs 


def get_doc_type_ids(sample_tokens:List[int],
                     doc_type_id:int) -> List:
    sample_tokens = np.array(sample_tokens)
    doc_type_id_list = list(np.where(sample_tokens == 1, doc_type_id, 0))
    return doc_type_id_list

def add_padding(sample_tokens:List[int],
                padding:bool,
                padding_num:int,
                max_length:Optional[int]) -> List:
    sample_tokens_len = len(sample_tokens)
    if padding:
        while len(sample_tokens_len) < max_length :
            sample_tokens += [padding_num] * (max_length-sample_tokens_len)
    return sample_tokens

    
