import os
import time
import torch
from dotenv import load_dotenv

from contextlib import contextmanager
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    HfArgumentParser
)
from datasets import load_dataset

from args import (
    ModelArguments,
    DataTrainingArguments,
    GenerationArguments
)

from utils.processor import preprocess_function
from utils.data_preprocessor import Preprocessor
from models.modeling_kobigbird_bart import EncoderDecoderModel

@contextmanager
def timer(name) :
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

def main() :
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, GenerationArguments)
    )
    model_args, data_args, generation_args = parser.parse_args_into_dataclasses()

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer
    )

    if model_args.use_model == "bigbart" :
        model = EncoderDecoderModel.from_pretrained(model_args.model_name_or_path)
    else :
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
        )
        model.config.output_attentions = True
    
    ### test 용 code ###
    load_dotenv(dotenv_path=data_args.use_auth_token_path)
    USE_AUTH_TOKEN = os.getenv("USE_AUTH_TOKEN")    
    
    dataset_name = "metamong1/summarization"
    datasets = load_dataset(dataset_name + "_part" if data_args.is_part else dataset_name, use_auth_token=USE_AUTH_TOKEN)
    data_preprocessor = Preprocessor()
    datasets = datasets.map(data_preprocessor.for_test)
    valid_dataset = datasets['validation']

    idx = 1600 ## 바꾸면서 test 해보세요!
    text = valid_dataset[idx]['text']
    title = valid_dataset[idx]['title']
    #####################
    # text = input("요약할 문장을 넣어주세요:")

    input_ids = tokenizer(text, add_special_tokens=True)
    
    if model_args.use_model != "bigbart" :
        input_ids = [tokenizer.bos_token_id] + input_ids['input_ids'][:-2] + [tokenizer.eos_token_id]
    else :
        input_ids = input_ids['input_ids']
    
    num_beams = data_args.num_beams
    if num_beams is not None :
        generation_args.num_return_sequences = num_beams

    with timer('** Generate title **') :
        summary_ids = model.generate(
            torch.tensor([input_ids]), num_beams=num_beams, **generation_args.__dict__)

        print('** text: ', text)
        print('** title: ', title)
        if len(summary_ids.shape) == 1  or summary_ids.shape[0] == 1:
            ## 출력 1개
            title = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
            print('Gen title 0', title)
        else :
            ## 출력 여러개
            titles = tokenizer.batch_decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
            for idx, title in enumerate(titles) :
                print('Gen title', idx, title)

if __name__ == "__main__":
    main()