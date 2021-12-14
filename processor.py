doc_type_dict = {
    "논문" : 0,
    "신문기사" : 1,
    "사설잡지" : 2
}

# label eos.id token 붙이기

def preprocess_function(examples, tokenizer, data_args):
    pad_id = tokenizer.pad_token_id
    doc_type_id = [doc_type_dict[doc_type] for doc_type in examples['doc_type']]
    
    prefix = tokenizer.bos_token
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    inputs = examples['text']
    titles = examples['title']
    inputs = [prefix + inp for inp in inputs]

    model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
    
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(titles, max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    if data_args.use_doc_type_ids:
        model_inputs = doc_type_marking(model_inputs, doc_type_id, pad_id)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def doc_type_marking(tokenizer_tmp_input, doc_type_id, pad_id):
    doc_type_input_ids=[]
    
    for i, input_ids_per_one_input in enumerate(tokenizer_tmp_input['input_ids']):
        if pad_id not in input_ids_per_one_input:
            marking = [doc_type_id[i]]*len(input_ids_per_one_input)
            doc_type_input_ids.append(marking)
        else:
            marking = [doc_type_id[i]]*len(input_ids_per_one_input[:list(input_ids_per_one_input).index(0)])
            tmp = [0] * len(input_ids_per_one_input[list(input_ids_per_one_input).index(0):])
            doc_type_input_ids.append(marking+tmp)
    tokenizer_tmp_input['doc_type_ids'] = doc_type_input_ids

    return tokenizer_tmp_input