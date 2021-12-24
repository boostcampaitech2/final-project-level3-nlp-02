# Korean Title Generator

Final Project in 2nd BoostCamp AI Tech 2기 by **메타몽팀 (2조)**

## Demo
![Demo](https://github.com/boostcampaitech2/final-project-level3-nlp-02/blob/dev/assets/%EC%8B%9C%EC%97%B0%EC%98%81%EC%83%81.gif?raw=true)


## Content
- [Project Abstract](#project-abstract)
- [How to use (최종 모델 checkpoint로 수정해야함)](#how-to-use--------checkpoint--------)
- [Result (결과 뽑고 수정 필요)](#result--------------)
- [Hardware](#hardware)
- [Operating System](#operating-system)
- [Archive Contents](#archive-contents)
- [Getting Started](#getting-started)
  * [Dependencies](#dependencies)
  * [Install Requirements](#install-requirements)
- [Arguments](#arguments)
  * [Model Arguments](#model-arguments)
  * [DataTrainingArguments](#datatrainingarguments)
  * [LoggingArguments](#loggingarguments)
  * [GenerationArguments](#generationarguments)
  * [Seq2SeqTrainingArguments](#seq2seqtrainingarguments)
- [Running Command](#running-command)
  * [Train](#train)
  * [Predict](#predict)
- [Reference](#reference)


## Project Abstract

🔥 생성 요약을 통한 한국어 문서 제목 생성기 🔥
- 데이터셋 :
  - 종류 : [논문 데이터](https://aihub.or.kr/aidata/30712) 162,341개, [문서 데이터](https://aihub.or.kr/aidata/8054) 371,290개
  - train_data : 275,219개 (Text, Title, Document Type)
  - validation_data : 91,741개 (Text, Title, Document Type)
  - test_data : 81,739개 (Text, Title, Document Type)

## How to use (최종 모델 checkpoint로 수정해야함)

``` python
import torch
from transformers import AutoConfig
from transformers import AutoTokenizer
from models.modeling_kobigbird_bart import EncoderDecoderModel

config = AutoConfig.from_pretrained('metamong1/bigbird-tapt-ep3')
tokenizer = AutoTokenizer.from_pretrained('metamong1/bigbird-tapt-ep3')
model = EncoderDecoderModel.from_pretrained('metamong1/bigbird-tapt-ep3', config=config)

text = "본 논문의 목적은 수도권 지역의 수출입 컨테이너 화물에 대한 최적 복합운송 네트워크를 찾는 데 있다. 따라서 이 지역의 컨테이너 화물의 물동량 흐름을 우선적으로 분석하였고, 총 수송비용과 총 수송 시간을 고려한 최적 경로를 구하려 시도하였다. 이를 위해 모형 설정은 0-1 이진변수를 이용한 목적계획법을 사용하였고, 유전알고리즘 기법을 통해 해를 도출하였다. 그 결과, 수도권 지역의 33개 각 시(군)에 대한 내륙 수송비용과 수송 시간을 최소화하는 수송거점 및 운송 수단을 도출함으로써 이 지역의 수출입 컨테이너 화물에 대한 최적 복합운송 네트워크를 발견할 수 있었다. 또한 시나리오별 수송비용 및 수송 시간의 절감 효과를 정량적으로 제시한다."

raw_input_ids = tokenizer.encode(text)
input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]

summary_ids = model.generate(torch.tensor([input_ids]))
tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
```

정답 제목
> 유전알고리즘을 이용한 복합운송최적화모형에관한 연구

생성 제목
> 유전알고리즘을 이용한 수도권의 수출입 컨테이너 화물에 대한 최적 복합운송



## Result (결과 뽑고 수정 필요)

|            | RougeL |
|:----------:|:------:|
| Evaluation | ------ |
|    Test    | ------ |


## Hardware

- Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz
- NVIDIA Tesla V100-SXM2-32GB

## Operating System

- Ubuntu 18.04.5 LTS

## Archive Contents

- final-project-level3-nlp-02 : 구현 코드와 모델 checkpoint 및 모델 결과를 포함하는 디렉토리

```
final-project-level3-nlp-02/
├── model
│   ├── args  
│   │   ├── __init__.py
│   │   ├── DataTrainingArguments.py
│   │   ├── GenerationArguments.py
│   │   ├── LoggingArguments.py
│   │   ├── ModelArguments.py
│   │   └── Seq2SeqTrainingArguments.py
│   ├── models  
│   │   ├── modeling_distilbert.py
│   │   ├── modeling_kobigbird_bart.py
│   │   └── modeling_longformer_bart.py
│   ├── utils   
│   │   ├── data_collator.py
│   │   ├── data_loader.py
│   │   ├── data_preprocessor.py
│   │   ├── rouge.py
│   │   └── trainer.py
│   ├── optimization   
│   │   ├── knowledge_distillation.py
│   │   ├── performance_test.py
│   │   ├── performnaceBenchmark.py
│   │   └── quantization.py
│   ├── predict.py
│   ├── pretrain.py
│   ├── REDAME.md
│   ├── running.sh
│   └── train.py
├── serving
│   ├── app.py
│   ├── GenerationArguments.py
│   ├── postprocessing.py
│   ├── predict.py
│   ├── utils.py
│   └── viz.py
├──.gitignore
└──requirements.sh

```

- `utils`
    - `utils/data_collator.py` : 모델에 입려되는 Batch를 생성하는 코드
    - `utils/data_preprocessor.py` : Text를 전처리하는 코드
    - `utils/processor.py` : Text를 Tokenizer를 이용해서 정수 인코딩을 하는 코드
    - `utils/rouge.py` : 모델의 평가지표와 관련되는 코드
    - `utils/trainer.py` : 모델을 학습하는데 활용하는 trainer 코드
- `models`
    - `modeling_kobigbird_bart.py` : bigbird bart 모델 코드
    - `modeling_longformerbart.py` : longformer bart 모델 코드
- `optimization`
    - `knowledge_distillation.py` :
    - `performanceBenchmark.py` : 
    - `performance_test.py` : 
    - `quantization.py` : 
- `predict.py` : 모델을 이용해서 입력된 문서의 제목을 생성하는 코드
- `pretrain.py` : summarization model을 fintune을 위한 코드
- `train.py` : summarization model을 pretrain을 위한 코드

## Getting Started

### Dependencies

- python=3.8.5
- transformers==4.11.0
- datasets==1.15.1
- torch==1.10.0
- streamlit==1.1.0
- elasticsearch==7.16.1
- pyvis==0.1.9
- plotly==5.4.0


### Install Requirements

```
sh requirements.sh
```

## Arguments
    
### Model Arguments

|   argument | description    | default         |
| :--------: | :----------------------- | :------------- |
|   model_name_or_path   | 사용할 사전 학습 모델 선택   | gogamza/kobart-base-v1   |
| use_model | 모델 타입 선택 [auto, bigbart, longbart] | auto |
|   config_name    | Pretrained된 model config 경로  | None |
|     tokenizer_name     | customized tokenizer 경로 선택   | None    |
| use_fast_tokenizer | fast tokenizer 사용 여부 | True |
| hidden_size | embedding hidden dimension 크기 | 128 |

👇 longBART specific
|      argument       | description   | default      |
| :-----------------: | :------------ | :----------- |
| attention_window_size | attention window 크기 | 256 |
| attention_head_size | attention head 개수 | 4 |
| encoder_layer_size | encoder layer 수 | 3 |
| decoder_layer_size | decoder layer 수 | 3 |



### DataTrainingArguments

|    argument  | description | default    |
| :-------: | :------------ | :------- |
| text_column | dataset에서 본문 column 이름 | text |
| title_column | dataset에서 제목 column 이름 | title |
| overwrite_cache |  캐시된 training과 evaluation set을 overwrite하기 | False |
| preprocessing_num_workers | 전처리동안 사용할 prcoess 수 지정 | 1 |
| max_source_length | Text Sequence 길이 지정 | 1024 |
| max_target_length | Title Sequence 길이 지정 | 128 |
| pad_to_max_length | 최대 길이로 Padding을 진행 | False |
| num_beams | Evaluation 할 때의 beam search에서 beam의 크기 | None |
| use_auth_token_path | Huggingface에 Dataset을 혹은 Model을 불러올 때 private key 주소 | ./use_auth_token.env |
| num_samples | train_dataset에서 sample 추출 갯수(None일 때는 전체 데이터 수 사용) | None |
| relative_eval_steps| Evaluation 횟수 | 10 |
| is_pretrain| Pretraining 여부 | False |
| is_part | 전체 데이터 수의 50% 정도 사용 | False |
| use_preprocessing | 전처리 여부 | False |
| use_doc_type_ids | doc_type_embedding 사용 여부 | False |


### LoggingArguments

|     argument          | description         | default        |
| :------:     | :-------------- | :------------- |
| wandb_unique_tag | wandb에 기록될 모델의 이름 | None |
| dotenv_path | wandb key값을 등록하는 파일의 경로  | ./wandb.env |
| project_name | wandb에 기록될 project name | Kobart |

### GenerationArguments

| argument | description  | default |
| :----: | :--------- | :------------- |
|    max_length        | 생성될 문장의 최대 길이    | None  |
| min_length    | 생성될 문장의 최소 길이  | 1 |
| length_penalty | 문장의 길이에 따라 주는 penalty의 정도 | 1.0 |
|    early_stopping        | Beam의 갯수 만큼 문장의 생성이 완료 되었을 때 생성을 종료 여부   | True  |
|    output_scores        | prediction score 출력 여부    | False  |
|    no_repeat_ngram_size        | 반복 생성되지 않을 ngram의 최소 크기 | 3  |
| num_return_sequences | 생성 문장 갯수 | 1 |
| top_k    | Top-K 필터링에서의 K 값  | 50 |
| top_p | 생성 과정에서 이어지는 토큰을 선택할 때의 최소 확률 값 | 0.95 |

### Seq2SeqTrainingArguments

|     argument  | description           | default        |
| :-----:  | :---------- | :------------- |
| metric_for_best_model  | train 후 저장될 모델 선정 기준 | rougeL  |
| es_patience  | early stopping이 되는 patience 값 | 3  |
| is_noam  | noam scheduler 사용 여부  | False |
| use_rdrop | R-drop 사용 여부 | False |
| reg_alpha | R-drop 사용 시 적용될 KL loss 비율 | 0.7 |
| alpha | knowledge distillation 시 CE loss 적용 비율 | 0.5 |
| temperature | distillation을 할 때의 temperature 값 | 1.0 |
| use_original | tiny distillation을 할 때 prediction loss 사용 여부 | False |
| teacher_check_point | teacher model의 checkpoint | None |
| use_teacher_forcing | teacher forcing 적용 여부 | False |


## Running Command
### Train
```
$ python train.py \
--model_name_or_path metamong1/bigbird-tapt-ep3 \
--use_model bigbart \
--do_train \
--output_dir checkpoint/kobigbirdbart_full_tapt_ep3_bs16_pre_noam \
--overwrite_output_dir \
--num_train_epochs 3 \
--use_doc_type_ids \
--max_source_length 2048 \
--max_target_length 128 \
--metric_for_best_model rougeLsum \
--es_patience 3 \
--load_best_model_at_end \
--project_name kobigbirdbart \
--wandb_unique_tag kobigbirdbart_full_tapt_ep5_bs16_pre_noam \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 16 \
--gradient_accumulation_steps 8 \
--use_preprocessing \
--warmup_steps 1000 \
--evaluation_strategy epoch \
--is_noam \
--learning_rate 0.08767941605644963 \
--save_strategy epoch
```

### Predict

```
$ python predict.py \
--model_name_or_path model/baseV1.0_Kobart_ep2_1210 \
--num_beams 3
```


## Reference
1. BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension
    > https://arxiv.org/pdf/1910.13461.pdf
2. Longformer: The Long-Document Transformer 
    > https://arxiv.org/pdf/2004.05150.pdf
3. Big Bird: Transformers for Longer Sequences
    > https://arxiv.org/pdf/2007.14062.pdf
4. Scheduled Sampling for Transformers 
    > https://arxiv.org/pdf/1906.07651.pdf
5. On the Effect of Dropping Layers of Pre-trained Transformer Models
    > https://arxiv.org/pdf/2004.03844.pdf
6. R-Drop: Regularized Dropout for Neural Networks
    > https://arxiv.org/pdf/2106.14448v2.pdf
