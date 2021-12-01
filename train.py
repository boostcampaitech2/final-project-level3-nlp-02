import os
import re
import torch
import random

import numpy as np
from dataloader import SumDataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed
)

from arguments import (
    ModelArguments,
    DataTrainingArguments,
    LoggingArguments,
)

from functools import partial
from datasets import load_dataset, load_metric

def rouge(hyp, ref, n):
    scores = []
    for h, r in zip(hyp, ref):
        r = re.sub(r'[UNK]', '', r)
        r = re.sub(r'[’!"#$%&\'()*+,-./:：？！《》;<=>?@[\\]^_`{|}~]+', '', r)
        r = re.sub(r'\d', '', r)
        r = re.sub(r'[a-zA-Z]', '', r)
        count = 0
        match = 0
        for i in range(len(ref) - n):
            gram = ref[i:i + n]
            if gram in hyp:
                match += 1
            count += 1
        scores.append(match / count)
    return np.average(scores)

def preprocess_function(examples, tokenizer, data_args):
    prefix = '[CLS]' # example
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    inputs = examples['text']
    targets = examples['title']
    inputs = [prefix + inp for inp in inputs]

    model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs



## 한번 손봐야함
def compute_metrics(eval_preds, tokenizer, data_args):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if data_args.ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    result = rouge(decoded_preds, decoded_labels)
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, LoggingArguments, TrainingArguments)
    )
    model_args, data_args, log_args, training_args = parser.parse_args_into_dataclasses()

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")
    
    set_seed(training_args.seed)

    
    train_dataset = SumDataset(data_args.dataset_name, 'train')
    valid_dataset = SumDataset(data_args.dataset_name, 'validation')

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cacher_dir ##
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        # from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir
    )
    

    print(config)
    print(tokenizer)
    print(model)


#  features = dataset.map(
#         prep_fn,
#         batched=is_batched,
#         num_proc=processor.data_args.preprocessing_num_workers,
#         remove_columns=dataset.column_names,
#         load_from_cache_file=not processor.data_args.overwrite_cache,
#     )
#     data_collator = DataCollatorForSeq2Seq(
#         tokenizer,
#         model=model,
#         label_pad_token_id=label_pad_token_id,
#         pad_to_multiple_of=8 if training_args.fp16 else None,
#     )




# model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
# tokenizer = AutoTokenizer.from_pretrained("t5-base")

# inputs = tokenizer("summarize: " + ARTICLE, return_tensors="pt", max_length=512, truncation=True)
# outputs = model.generate(
#     inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True
# )

# print(tokenizer.decode(outputs[0]))

# prep_fn  = partial(preprocess_function, tokenizer=tokenizer, data_args=data_args)
# comp_met_fn  = partial(compute_metrics, tokenizer=tokenizer, data_args=data_args)

# if __name__ == "__main__":