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

from models.modeling_longformerbart import LongformerBartConfig, LongformerBartWithDoctypeForConditionalGeneration
from models.modeling_kobigbird_bart import (
    EncoderDecoderModel, 
    BigBirdConfigWithDoctype, 
    BartConfigWithDoctype, 
    BigBirdModelWithDoctype, 
    BartDecoderWithDoctype
)

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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    
    seed_everything(training_args.seed)
    if training_args.do_eval :
        training_args.predict_with_generate = True
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
    datasets = load_dataset(dataset_name + "_part" if data_args.is_part else dataset_name, use_auth_token=USE_AUTH_TOKEN)
    data_preprocessor = Preprocessor()
    data_filter = Filter(min_size=5, max_size=100)

    ## data preprocessing
    datasets = datasets.map(data_preprocessor.for_train)
    datasets = datasets.filter(data_filter)

    train_dataset = datasets['train']
    valid_dataset = datasets['validation']

    if data_args.num_samples:
        train_dataset = train_dataset.select(range(data_args.num_samples))
        valid_dataset = valid_dataset.select(range(data_args.num_samples))

    train_dataset.cleanup_cache_files()
    valid_dataset.cleanup_cache_files()

    train_dataset = train_dataset.shuffle(training_args.seed)
    valid_dataset = valid_dataset.shuffle(training_args.seed)
    print('** Dataset example')
    print(f"[for Train Dataset] : {train_dataset[0]['title']}")
    print(f"[for Valid Dataset] : {valid_dataset[0]['title']}")

    column_names = train_dataset.column_names
    if data_args.relative_eval_steps :
        # Train 동안 relative_eval_steps count 회수 만큼 evaluation 
        # 전체 iteration에서 eval 횟수로 나누어 evaluation step
        iter_by_epoch = math.ceil(len(train_dataset)/(training_args.per_device_train_batch_size*training_args.gradient_accumulation_steps))
        training_args.num_training_steps =  iter_by_epoch * training_args.num_train_epochs
        training_args.eval_steps = int(training_args.num_training_steps // data_args.relative_eval_steps)
        training_args.save_steps = training_args.eval_steps # save step은 eval step의 배수여야 함

    print(f"train_dataset length: {len(train_dataset)}")
    print(f"valid_dataset length: {len(valid_dataset)}")
    print(f"eval_steps: {training_args.eval_steps}")

    
    # model별 config 호출
    if model_args.use_model == "longbart":
        config = LongformerBartConfig.from_pretrained(
                model_args.config_name if model_args.config_name else model_args.model_name_or_path)

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
    
    elif model_args.use_model=="bigbart":
        config = {}
        config["encoder"] = BigBirdConfigWithDoctype.from_pretrained("monologg/kobigbird-bert-base")
        config["decoder"] = BartConfigWithDoctype.from_pretrained("gogamza/kobart-base-v1")
        
        config["encoder"].encoder_layers = 6
        config["decoder"].vocab_size = config["encoder"].vocab_size
        config["decoder"].pad_token_id = config["encoder"].pad_token_id
        config["decoder"].max_position_embeddings = data_args.max_target_length
        if training_args.use_teacher_forcing :
            config["decoder"].num_training_steps = training_args.num_training_steps
        training_args.model_config = config["decoder"]

        if data_args.use_doc_type_ids :
            config["encoder"].doc_type_size = 3
            config["decoder"].doc_type_size = 3
    else :
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer
    )
    
    def model_init():
        if model_args.use_model == "longbart":
            return LongformerBartWithDoctypeForConditionalGeneration.from_pretrained(model_args.model_name_or_path, training_args.num_training_steps)
        elif model_args.use_model == "bigbart":
            # https://discuss.huggingface.co/t/fixing-the-random-seed-in-the-trainer-does-not-produce-the-same-results-across-runs/3442
            # Producibility parameter initialization
            encoder = BigBirdModelWithDoctype.from_pretrained("monologg/kobigbird-bert-base",config=config["encoder"])
            decoder = BartDecoderWithDoctype.from_pretrained("gogamza/kobart-base-v1", config=config["decoder"])
            
            for i in range(1,6):
                encoder.encoder.layer[i] = encoder.encoder.layer[2*i]
            encoder.encoder.layer = encoder.encoder.layer[:config["encoder"].encoder_layers]
            decoder.embed_tokens = encoder.embeddings.word_embeddings
            return EncoderDecoderModel(encoder = encoder, decoder = decoder)     
        else :
            return AutoModelForSeq2SeqLM.from_pretrained(
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
    pad_to_multiple_of = model_args.attention_window_size if model_args.use_model=="longbart" else (
        8 if training_args.fp16 else None
    )
    data_collator = DataCollatorForSeq2SeqWithDocType(
        tokenizer,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=pad_to_multiple_of
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
    
    if training_args.distillation_type == 'distil':
        print('DistillationTrainer is used!!!')
        teacher_config = AutoConfig.from_pretrained(training_args.teacher_check_point)
        teacher_model=AutoModelForSeq2SeqLM.from_pretrained(training_args.teacher_check_point, config=teacher_config).to(device)
        trainer = DistillationTrainer(
            args=training_args,
            teacher_model = teacher_model,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=comp_met_fn if training_args.predict_with_generate else None,
            model_init=model_init, ## model 성능 재현
            callbacks = [EarlyStoppingCallback(early_stopping_patience=training_args.es_patience)] if training_args.es_patience else None
        )
    elif training_args.distillation_type == 'tiny':
        print('TinyTrainer is used!!!')
        teacher_config = AutoConfig.from_pretrained(training_args.teacher_check_point)
        teacher_model=AutoModelForSeq2SeqLM.from_pretrained(training_args.teacher_check_point, config=teacher_config).to(device)
        trainer = TinyTrainer(
            args=training_args,
            teacher_model = teacher_model,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=comp_met_fn if training_args.predict_with_generate else None,
            model_init=model_init, ## model 성능 재현
            callbacks = [EarlyStoppingCallback(early_stopping_patience=training_args.es_patience)] if training_args.es_patience else None
        )
    else:
        trainer = Seq2SeqTrainerWithConditionalDocType(
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