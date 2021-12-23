from dotenv import load_dotenv
import wandb

import os
import math
import random
import numpy as np

import torch
import torch.nn as nn

from functools import partial
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    HfArgumentParser,
    EarlyStoppingCallback
)

from args import (
    DataTrainingArguments,
    LoggingArguments,
    ModelArguments,
    CustomSeq2SeqTrainingArguments
)

from utils.trainer import Seq2SeqTrainerWithConditionalDocType
from utils.data_preprocessor import Preprocessor, Filter
from utils.data_collator import DataCollatorForSeq2SeqWithDocType
from utils.processor import preprocess_function
from utils.rouge import compute_metrics
from optimization.knowledge_distillation import DistillationTrainer, TinyTrainer
from transformers.models.distilbert.configuration_distilbert import DistilBertConfig
from transformers import DistilBertTokenizerFast
from models.modeling_distilbert_bart import DistilBertForConditionalGeneration
from models.modeling_longformer_bart import LongformerBartConfig, LongformerBartWithDoctypeForConditionalGeneration
from models.modeling_kobigbird_bart import EncoderDecoderModel

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
def main():
    ## Arguments setting
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, LoggingArguments, CustomSeq2SeqTrainingArguments)
    )
    model_args, data_args, log_args, training_args = parser.parse_args_into_dataclasses()    
    
    seed_everything(training_args.seed)
    assert training_args.do_eval
    training_args.predict_with_generate = True
    print(f"** Test mode: { training_args.do_eval}")
    print(f"** model is from {model_args.model_name_or_path}")

        ## load and process dataset    
    load_dotenv(dotenv_path=data_args.use_auth_token_path)
    USE_AUTH_TOKEN = os.getenv("USE_AUTH_TOKEN")    
    test_dataset = load_dataset('metamong1/summarization', split="test", use_auth_token=USE_AUTH_TOKEN)

    if data_args.use_preprocessing:
        data_preprocessor = Preprocessor()
        test_dataset = test_dataset.map(data_preprocessor.for_test)
        data_args.preprocessing_num_workers = 1


    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer
    )
    
    # model 호출
    if "bigbart" in model_args.use_model :
        model = EncoderDecoderModel.from_pretrained(model_args.model_name_or_path)
    elif "longbart" in model_args.use_model:
        model = LongformerBartWithDoctypeForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path)
    
    prep_fn  = partial(preprocess_function, tokenizer=tokenizer, data_args=data_args)
    column_names = test_dataset.column_names
    test_dataset = test_dataset.map(
        prep_fn,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on train dataset",
    )

    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if model_args.use_model=="longbart":
        pad_to_multiple_of = model_args.attention_window_size
    elif data_args.use_preprocessing:
        pad_to_multiple_of = 1
    elif training_args.fp16:
        pad_to_multiple_of = 8
    else:
        pad_to_multiple_of = None
        
    data_collator = DataCollatorForSeq2SeqWithDocType(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=pad_to_multiple_of
    )

    # wandb
    load_dotenv(dotenv_path=log_args.dotenv_path)
    WANDB_AUTH_KEY = os.getenv("WANDB_AUTH_KEY")
    wandb.login(key=WANDB_AUTH_KEY)

    wandb.init(
        entity="final_project",
        project="test",
        name=log_args.wandb_unique_tag
    )
    wandb.config.update(training_args)
    
    comp_met_fn  = partial(compute_metrics, tokenizer=tokenizer, data_args=data_args)
    
    trainer = Seq2SeqTrainerWithConditionalDocType(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=comp_met_fn if training_args.predict_with_generate else None,
    )

    max_length = (training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length)
    results = {}
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
    print("#########Eval metrics: #########", metrics) 
    metrics["eval_samples"]=len(test_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    return results
    
if __name__ == "__main__":
    main()