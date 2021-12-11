
import numpy as np

doc_type_dict = {
    "논문" : 1,
    "신문기사" : 2,
    "사설잡지" : 3,
}

def preprocess_function(examples, tokenizer, data_args):
    bos_token = tokenizer.bos_token
    eos_token = tokenizer.eos_token
    padding = "max_length" if data_args.pad_to_max_length else False

    inputs = examples['text']
    doc_types = examples['doc_type']
    inputs = [bos_token + inp + eos_token for inp in inputs]

    model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
    doc_type_ids = []
    for i in range(len(model_inputs['input_ids'])) :
        attn_mask = model_inputs['attention_mask'][i]
        attn_mask = np.array(attn_mask)

        doc_type_id_list = list(np.where(attn_mask == 1, doc_type_dict[doc_types[i]], 0))
        doc_type_ids.append(doc_type_id_list)

    model_inputs['doc_type_ids'] = doc_type_ids
    return model_inputs