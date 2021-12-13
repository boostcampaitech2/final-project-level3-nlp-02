import os
import importlib.util
import random
import math
from dotenv import load_dotenv

import numpy as np
import torch
import torch.nn as nn
import wandb

from functools import partial
from transformers import (
    HfArgumentParser,
    EarlyStoppingCallback
)

from datasets import load_dataset

from args import (
    DataTrainingArguments,
    LoggingArguments,
    ModelArguments,
    CustomSeq2SeqTrainingArguments,
)

from dataloader import SumDataset
from processor import preprocess_function
from rouge import compute_metrics

from trainer import Seq2SeqTrainerWithDocType

from models.rebuilding_longformerbart import make_model_for_changing_postion_embedding
from models.modeling_longformerbart import (
    LongformerBartConfig,
    LongformerBartWithDoctypeForConditionalGeneration,
    )
from data_collator import (
    DataCollatorForSeq2SeqWithDocType,
    DataCollatorForTextInfillingDocType
    )

from utils.tokenizer import BartTokenizerWithDocType

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
    training_args.model_path = f"{model_args.longformerbart_path}_{data_args.max_source_length}_{data_args.max_target_length}"
    seed_everything(training_args.seed)
    print(f"** Train mode: { training_args.do_train}")
    print(f"** model is from {model_args.model_name_or_path}")
    print(f'** max_target_length:', data_args.max_target_length)


    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )

    ## load and process dataset
    load_dotenv(dotenv_path=data_args.use_auth_token_path)
    USE_AUTH_TOKEN = os.getenv("USE_AUTH_TOKEN")

    dataset_name = "metamong1/summarization"
    train_dataset = load_dataset(dataset_name+"_part" if data_args.is_part else dataset_name,
                            split="train",
                            use_auth_token=USE_AUTH_TOKEN)
    if data_args.num_samples is not None:
        train_dataset = train_dataset.select(range(data_args.num_samples))
    train_dataset.cleanup_cache_files()

    train_dataset = train_dataset.shuffle(training_args.seed)
    print('** Dataset example', train_dataset[0]['title'], train_dataset[0]['title'], sep = '\n')

    column_names = train_dataset.column_names
    print(f"train_dataset length: {len(train_dataset)}")

    config = LongformerBartConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path)
    
    # if not os.path.exists(training_args.model_path):
    #     make_model_for_changing_postion_embedding(config,data_args,model_args)

    config.encoder_layers = model_args.encoder_layer_size
    config.decoder_layers = model_args.decoder_layer_size
    config.d_model = model_args.hidden_size
    config.encoder_attention_heads = model_args.attention_head_size
    config.decoder_attention_heads = model_args.attention_head_size
    config.max_position_embeddings = data_args.max_source_length
    config.max_target_positions = data_args.max_target_length
    config.attention_window = [model_args.attention_window_size]*model_args.encoder_layer_size
    config.attention_dropout = model_args.dropout
    config.dropout = model_args.dropout

    tokenizer = BartTokenizerWithDocType.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer
    )
    training_args.model_config = config
    def model_init(training_args):
        # https://discuss.huggingface.co/t/fixing-the-random-seed-in-the-trainer-does-not-produce-the-same-results-across-runs/3442
        # Producibility parameter initialization
        model = LongformerBartWithDoctypeForConditionalGeneration._from_config(training_args.model_config)
        return model
        
    prep_fn  = partial(preprocess_function, tokenizer=tokenizer, data_args=data_args)
    train_dataset = train_dataset.map(
        prep_fn,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on train dataset",
    )

    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.is_pretrain:
        data_collator = DataCollatorForTextInfillingDocType(
            tokenizer,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=model_args.attention_window_size,
        )
    else:    
        data_collator = DataCollatorForSeq2SeqWithDocType(
            tokenizer,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=model_args.attention_window_size,
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
    
    # comp_met_fn  = partial(compute_metrics, tokenizer=tokenizer, data_args=data_args)
    
    trainer = Seq2SeqTrainerWithDocType(
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        config=config,
        # compute_metrics=comp_met_fn if training_args.predict_with_generate else None,
        model_init=model_init,
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

    
if __name__ == "__main__":
    main()