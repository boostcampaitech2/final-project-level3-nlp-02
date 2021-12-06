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

from rdrop_trainer import RdropTrainer
from dataloader import SumDataset
from transformers.trainer_utils import get_last_checkpoint
from processor import preprocess_function
from rouge import compute_metrics

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, LoggingArguments, CustomSeq2SeqTrainingArguments)
    )
    model_args, data_args, log_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.do_eval :
        training_args.predict_with_generate = True

    print(f"** Train mode: { training_args.do_train}")
    print(f"** model is from {model_args.model_name_or_path}")
    print(f"** data is from {data_args.dataset_name}")
    print(f'** max_target_length:', data_args.max_target_length)

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
    seed_everything(training_args.seed)

    types = data_args.dataset_name.split(',')
    data_args.dataset_name = ['metamong1/summarization_' + dt for dt in types]
    
    load_dotenv(dotenv_path=data_args.use_auth_token_path)
    USE_AUTH_TOKEN = os.getenv("USE_AUTH_TOKEN")
    
    train_dataset = SumDataset(data_args.dataset_name, 'train', USE_AUTH_TOKEN=USE_AUTH_TOKEN).load_data()
    valid_dataset = SumDataset(data_args.dataset_name, 'validation', USE_AUTH_TOKEN=USE_AUTH_TOKEN).load_data()
    train_dataset.cleanup_cache_files()
    valid_dataset.cleanup_cache_files()
    
    column_names = train_dataset.column_names
    if training_args.do_train and training_args.do_eval:
        sampler_cnt = data_args.max_eval_samples*len(data_args.dataset_name)
        valid_dataset = valid_dataset.shuffle(seed=training_args.seed).select(range(sampler_cnt)) ## validation sampler, default: 10000*dataset_count

    if training_args.relative_eval_steps :
        iterations =  training_args.num_train_epochs*math.ceil(len(train_dataset)/training_args.per_device_train_batch_size)
        training_args.eval_steps = int(iterations // training_args.relative_eval_steps) ## dataset 크기에 상대적 eval step 적용
        training_args.save_steps = training_args.eval_steps

    print(f"train_dataset length: {len(train_dataset)}")
    print(f"valid_dataset length: {len(valid_dataset)}")
    print(f"eval_steps: {training_args.eval_steps}")

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir
    )
    
    prep_fn  = partial(preprocess_function, tokenizer=tokenizer, data_args=data_args)
    train_dataset = train_dataset.map(
        prep_fn,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on train dataset",
    )

    valid_dataset = valid_dataset.map(
        prep_fn,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on validation dataset",
    )

    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
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

    if model_args.use_rdrop:
        trainer = RdropTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset, # if training_args.do_train else None,
            eval_dataset=valid_dataset, # if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=comp_met_fn if training_args.predict_with_generate else None,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=training_args.es_patience)] if training_args.es_patience else None
        )
    else:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset, # if training_args.do_train else None,
            eval_dataset=valid_dataset, # if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=comp_met_fn if training_args.predict_with_generate else None,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=training_args.es_patience)] if training_args.es_patience else None
        )

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
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