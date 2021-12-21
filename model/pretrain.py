import os
import math
import wandb
import torch
import random
import numpy as np
from dotenv import load_dotenv


from functools import partial
from transformers import HfArgumentParser
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from utils.processor import preprocess_function
from utils.trainer import Seq2SeqTrainerWithConditionalDocType
from utils.data_collator import DataCollatorForTextInfillingDocType

from args import (
    DataTrainingArguments,
    LoggingArguments,
    ModelArguments,
    CustomSeq2SeqTrainingArguments,
)

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
    training_args.model_path = f"{model_args.longformerbart_path}_{data_args.max_source_length}_{data_args.max_target_length}"
    seed_everything(training_args.seed)
    print(f"** Train mode: { training_args.do_train}")
    print(f"** model is from {model_args.model_name_or_path}")
    print(f'** max_target_length:', data_args.max_target_length)

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

    if model_args.use_model == "longbart" :
        config = LongformerBartConfig.from_pretrained(model_args.config_name if model_args.config_name else model_args.model_name_or_path)
        config.doc_type_size = 4 if data_args.use_doc_type_ids else -1
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
        config.encoder_ffn_dim = config.d_model*4
        config.decoder_ffn_dim = config.d_model*4
        
    if model_args.use_model == "bigbart":
        config = {}
        config["encoder"] = BigBirdConfigWithDoctype.from_pretrained("monologg/kobigbird-bert-base")
        config["decoder"] = BartConfigWithDoctype.from_pretrained("gogamza/kobart-base-v1")
        
        config["encoder"].num_hidden_layers = 6
        config["decoder"].vocab_size = config["encoder"].vocab_size
        config["decoder"].pad_token_id = config["encoder"].pad_token_id
        config["decoder"].max_position_embeddings = data_args.max_target_length
        if training_args.use_teacher_forcing :
            config["decoder"].num_training_steps = training_args.num_training_steps
        training_args.model_config = config["decoder"]

        if data_args.use_doc_type_ids :
            config["encoder"].doc_type_size = 4
            config["decoder"].doc_type_size = 4

    if training_args.use_teacher_forcing:
        iter_by_epoch = math.ceil(len(train_dataset)/training_args.per_device_train_batch_size)
        config.num_training_steps =  iter_by_epoch * training_args.num_train_epochs
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer
    )

    training_args.model_config = config
    def model_init():
        if model_args.use_model == "longbart":
            model = LongformerBartWithDoctypeForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
            model.config.dropout = model_args.dropout
            model.config.attention_dropout = model_args.dropout
            model.num_training_steps = config.num_training_steps
            return model
        elif model_args.use_model == "bigbart":
            encoder = BigBirdModelWithDoctype.from_pretrained("monologg/kobigbird-bert-base",config=config["encoder"])
            decoder = BartDecoderWithDoctype.from_pretrained("gogamza/kobart-base-v1", config=config["decoder"])
            decoder.embed_tokens = encoder.embeddings.word_embeddings
            model =  EncoderDecoderModel(encoder = encoder, decoder = decoder)   
            return model 
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

    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    pad_to_multiple_of = model_args.attention_window_size if model_args.use_model=="longbart" else (
        8 if training_args.fp16 else None
    )

    data_collator = DataCollatorForTextInfillingDocType(
        tokenizer,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    # # wandb
    # load_dotenv(dotenv_path=log_args.dotenv_path)
    # WANDB_AUTH_KEY = os.getenv("WANDB_AUTH_KEY")
    # wandb.login(key=WANDB_AUTH_KEY)

    # wandb.init(
    #     entity="final_project",
    #     project=log_args.project_name,
    #     name=log_args.wandb_unique_tag
    # )
    # wandb.config.update(training_args)

    trainer = Seq2SeqTrainerWithConditionalDocType(
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        model_init=model_init,
    )

    if training_args.do_train:
        last_checkpoint = None
        if (
            os.path.isdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
        ):

            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )

        
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        print("#########Train result: #########", train_result)

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    
if __name__ == "__main__":
    main()