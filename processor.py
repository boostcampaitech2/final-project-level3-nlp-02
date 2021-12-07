def preprocess_function(examples, tokenizer, data_args):
    prefix = tokenizer.bos_token # example
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

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs