import os
import random
import math
from dotenv import load_dotenv

import numpy as np
import torch
import wandb

from functools import partial
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    HfArgumentParser,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)

from args import (
    DataTrainingArguments,
    LoggingArguments,
    ModelArguments,
    CustomSeq2SeqTrainingArguments
)

from trainer import Seq2SeqTrainerWithDocType
from models.bart_doctype import BartWithDocTypeConfig, BartWithDocTypeForConditionalGeneration

from dataloader import SumDataset
from processor import preprocess_function
from rouge import compute_metrics
from preprocessor import Filter
from data_collator import DataCollatorForSeq2SeqWithDocType

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
    if training_args.do_eval :
        training_args.predict_with_generate = True
    print(f"** Train mode: { training_args.do_train}")
    print(f"** model is from {model_args.model_name_or_path}")
    print(f"** data is from {data_args.dataset_name}")
    print(f'** max_target_length:', data_args.max_target_length)

    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )
    
    ## load and process dataset
    types = data_args.dataset_name.split(',')
    data_args.dataset_name = ['metamong1/summarization_' + dt for dt in types]
    
    load_dotenv(dotenv_path=data_args.use_auth_token_path)
    USE_AUTH_TOKEN = os.getenv("USE_AUTH_TOKEN")

    train_dataset = SumDataset(
        data_args.dataset_name,
        'train',
        shuffle_seed=training_args.seed,
        ratio=data_args.relative_sample_ratio,
        USE_AUTH_TOKEN=USE_AUTH_TOKEN
    ).load_data()

    valid_dataset = SumDataset(
        data_args.dataset_name,
        'validation',
        shuffle_seed=training_args.seed,
        ratio=data_args.relative_sample_ratio,
        USE_AUTH_TOKEN=USE_AUTH_TOKEN
    ).load_data()
    
    train_dataset.cleanup_cache_files()
    valid_dataset.cleanup_cache_files()

    train_dataset = train_dataset.shuffle(training_args.seed)
    valid_dataset = valid_dataset.shuffle(training_args.seed)
    print('** Dataset example', train_dataset[0]['title'], train_dataset[1]['title'], sep = '\n')

    column_names = train_dataset.column_names
    if data_args.relative_eval_steps :
        ## Train 동안 relative_eval_steps count 회수 만큼 evaluation 
        ## 전체 iteration에서 eval 횟수로 나누어 evaluation step
        iter_by_epoch = math.ceil(len(train_dataset)/training_args.per_device_train_batch_size)
        iterations =  iter_by_epoch * training_args.num_train_epochs
        training_args.eval_steps = int(iterations // data_args.relative_eval_steps)
        training_args.save_steps = training_args.eval_steps ## save step은 eval step의 배수여야 함

    # data_filter = Filter(min_size=5, max_size=80)
    # train_dataset = train_dataset.filter(data_filter)
    # valid_dataset = valid_dataset.filter(data_filter)

    print(f"train_dataset length: {len(train_dataset)}")
    print(f"valid_dataset length: {len(valid_dataset)}")
    print(f"eval_steps: {training_args.eval_steps}")

    config = BartWithDocTypeConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        doc_type_size=3+1, # document size + padding 
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer
    )

    def model_init():
        # https://discuss.huggingface.co/t/fixing-the-random-seed-in-the-trainer-does-not-produce-the-same-results-across-runs/3442
        # Producibility parameter initialization
        return BartWithDocTypeForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config
        )

    prep_fn  = partial(preprocess_function, tokenizer=tokenizer, data_args=data_args)
    train_dataset = train_dataset.map(
        prep_fn,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on train dataset",
    )

    valid_dataset = valid_dataset.map(
        prep_fn,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on validation dataset",
    )

    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2SeqWithDocType(
        tokenizer,
        padding=True,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # wandb
    load_dotenv(dotenv_path=log_args.dotenv_path)
    WANDB_AUTH_KEY = os.getenv("WANDB_AUTH_KEY")
    wandb.login(key=WANDB_AUTH_KEY)

    wandb.init(
        entity="final_project",
        project=log_args.project_name,
        name=log_args.wandb_unique_tag
    )
    wandb.config.update(training_args)
    
    comp_met_fn  = partial(compute_metrics, tokenizer=tokenizer, data_args=data_args)
    trainer = Seq2SeqTrainerWithDocType(
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=comp_met_fn if training_args.predict_with_generate else None,
        model_init=model_init, ## model 성능 재현
        callbacks = [EarlyStoppingCallback(early_stopping_patience=training_args.es_patience)] if training_args.es_patience else None
    )

    if training_args.do_train:
        train_result = trainer.train()
        print("#########Train result: #########", train_result)
        trainer.save_model()

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    max_length = (training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length)
    results = {}
    
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    if not training_args.do_train and training_args.do_eval:


        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        print("#########Eval metrics: #########", metrics) 
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(valid_dataset)
        metrics["eval_samples"]=min(max_eval_samples, len(valid_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    return results
    
if __name__ == "__main__":
    main()