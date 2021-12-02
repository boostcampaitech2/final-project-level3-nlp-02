import time
import torch

from contextlib import contextmanager
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    HfArgumentParser
)

from arguments import (
    ModelArguments,
    DataTrainingArguments,
    GenArguments
)

@contextmanager
def timer(name) :
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

def main() :
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, GenArguments)
    )
    model_args, data_args, generation_args = parser.parse_args_into_dataclasses()

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir ##
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

    # text = input("요약할 문장을 넣어주세요:")
    text = "과거를 떠올려보자. 방송을 보던 우리의 모습을. 독보적인 매체는 TV였다. 온 가족이 둘러앉아 TV를 봤다. 간혹 가족들끼리 뉴스와 드라마, 예능 프로그램을 둘러싸고 리모컨 쟁탈전이 벌어지기도  했다. 각자 선호하는 프로그램을 ‘본방’으로 보기 위한 싸움이었다. TV가 한 대인지 두 대인지 여부도 그래서 중요했다. 지금은 어떤가. ‘안방극장’이라는 말은 옛말이 됐다. TV가 없는 집도 많다. 미디어의 혜 택을 누릴 수 있는 방법은 늘어났다. 각자의 방에서 각자의 휴대폰으로, 노트북으로, 태블릿으로 콘텐츠 를 즐긴다."

    raw_input_ids =  tokenizer(text, max_length=data_args.max_source_length, truncation=True)
    input_ids = [tokenizer.bos_token_id] + raw_input_ids['input_ids'][:-2] + [tokenizer.eos_token_id]

    num_beams=data_args.num_beams
    with timer('** Generate title **') :
        summary_ids = model.generate(torch.tensor([input_ids]), num_beams=num_beams, **generation_args.__dict__)
        if len(summary_ids.shape) == 1 :
            title = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
            print(title)
        else :
            titles = tokenizer.batch_decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
            print(titles)

if __name__ == "__main__":
    main()