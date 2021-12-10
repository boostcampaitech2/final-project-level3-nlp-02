
import numpy as np

doc_type_dict = {
    "논문" : 4,
    "신문기사" : 5,
    "사설잡지" : 6
}
def preprocess_function(examples, tokenizer, data_args):
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    max_source_length = data_args.max_source_length
    max_target_length = data_args.max_target_length
    padding = "max_length"# if data_args.pad_to_max_length else False
    inputs = examples['text']
    titles = examples['title']
    doc_type_id = [doc_type_dict[doc_type] for doc_type in examples['doc_type']]
    
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)
    pad_idx_lists = [np.where(np.array(input_idx_list)==3) for input_idx_list in model_inputs['input_ids']]
    encoder_doc_type_ids_list = []
    for i, pad_idx_list in enumerate(pad_idx_lists):
        pad_idx_list = pad_idx_list[0]
        if len(pad_idx_list) > 1:
            pad_idx = pad_idx_list[0]
            model_inputs['input_ids'][i].insert(pad_idx, eos_id)
            model_inputs['input_ids'][i] = [bos_id] + model_inputs['input_ids'][i][:max_source_length-1]
            model_inputs['attention_mask'][i][:pad_idx+2] = [1]*(pad_idx+2)
            model_inputs['attention_mask'][i] = model_inputs['attention_mask'][i][:max_source_length]
        else:
            model_inputs['input_ids'][i] = [bos_id] + model_inputs['input_ids'][i][:max_source_length-2] + [eos_id]
        encoder_doc_type_ids = [doc_type_id[i] if id_ == 1 else id_ for id_ in model_inputs['attention_mask'][i]]
        encoder_doc_type_ids_list.append(encoder_doc_type_ids)
    

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(titles, max_length=max_target_length, padding=padding, truncation=True)
    
    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(token_id if token_id != tokenizer.pad_token_id else -100) for token_id in label] for label in labels["input_ids"]
        ]
    
    pad_idx_lists = [np.where(np.array(label)== -100) for label in labels['input_ids']]
    # print(pad_idx_list[0][0])
    decoder_doc_type_ids_list = []
    for i, pad_idx_list in enumerate(pad_idx_lists):
        pad_idx_list = pad_idx_list[0]
        if len(pad_idx_list) > 1:
            pad_idx = pad_idx_list[0]
            labels['input_ids'][i].insert(pad_idx, eos_id)
            labels['input_ids'][i] = [bos_id] + labels['input_ids'][i][:max_target_length-1]
            labels['attention_mask'][i][:pad_idx+2] = [1]*(pad_idx+2)
            labels['attention_mask'][i] = labels['attention_mask'][i][:max_target_length]
        else:
            labels['input_ids'][i] = [bos_id] + labels['input_ids'][i][:max_target_length-2] + [eos_id]
        decoder_doc_type_ids = [doc_type_id[i] if id_ == 1 else id_ for id_ in labels['attention_mask'][i]]
        decoder_doc_type_ids_list.append(decoder_doc_type_ids)
    
    model_inputs["labels"] = labels['input_ids']
    model_inputs['decoder_attention_mask'] = labels['attention_mask']
    model_inputs['encoder_doc_type_ids'] = encoder_doc_type_ids_list
    model_inputs['decoder_doc_type_ids'] = decoder_doc_type_ids_list
    
    return model_inputs 