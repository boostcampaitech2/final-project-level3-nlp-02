import numpy as np

doc_type_dict = {
    "논문" : 1,
    "신문기사" : 2,
    "사설잡지" : 3,
}

def preprocess_function(examples, tokenizer, data_args):
    prefix = tokenizer.bos_token
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    max_source_length = data_args.max_source_length
    max_target_length = data_args.max_target_length
    padding = False

    inputs = examples['text']
    titles = examples['title']
    doc_types = examples['doc_type']

    inputs = [prefix + inp for inp in inputs]

    model_inputs = tokenizer(inputs, max_length=max_source_length-1, padding=padding, truncation=True)
    
    doc_type_ids = []
    for i in range(len(model_inputs['input_ids'])) :
        model_inputs["attention_mask"][i] = model_inputs["attention_mask"][i]+[1]
        model_inputs["input_ids"][i] = model_inputs["input_ids"][i]+[eos_token_id]
        model_inputs["token_type_ids"][i] = model_inputs["token_type_ids"][i]+[0]

        doc_type_id = doc_type_dict[doc_types[i]]
        attn_mask = np.array(model_inputs['attention_mask'][i])
        doc_type_id_list = list(np.where(attn_mask == 1, doc_type_id, 0))
        doc_type_ids.append(doc_type_id_list)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(titles, max_length=max_target_length, padding=padding, truncation=True)

    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["doc_type_ids"] = doc_type_ids
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs