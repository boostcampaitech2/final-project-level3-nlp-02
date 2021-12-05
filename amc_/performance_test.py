import datasets
from dotenv import load_dotenv
import transformers
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
import torch

import os
import argparse

from performanceBenchmark import PerformanceBenchmark

load_dotenv(verbose=True)

def main(args):
    # 기본 세팅
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    api_token = os.getenv('HF_DATASET_API_TOKEN')
    TEST_DATASET_SIZE = args.test_dataset_size
    check_point = args.checkpoint  # 'kobart-summarization-finetuned-paper-sample-size-1000/checkpoint-1000'

    if args.cpu_flag:
        device='cpu'

    # 데이터셋 준비
    dataset = datasets.load_dataset(args.test_dataset, use_auth_token=api_token)
    test_dataset = dataset['validation'].shuffle().filter(lambda x: len(x['text'])< 500).select(range(TEST_DATASET_SIZE))

    # 토크나이저 준비
    tokenizer = AutoTokenizer.from_pretrained(check_point)

    # 모델 준비
    model = AutoModelForSeq2SeqLM.from_pretrained(check_point).to(device)

    # 사용할 모델 및 파이프라인 준비
    summerizer = pipeline(
        'summarization', 
        model=model,
        tokenizer=tokenizer,
        device = 0 if torch.cuda.is_available() and not args.cpu_flag else -1
    )

    # 벤치마크 준비
    performance_benchmark = PerformanceBenchmark(summerizer, test_dataset, tokenizer, 'baseline')

    # 벤치마크 계산
    test_categories = args.test_categories.split(',')
    if 'rouge' in test_categories:
        performance_benchmark.compute_rouge()
    
    if 'size' in test_categories:
        performance_benchmark.compute_size()
    
    if 'time' in test_categories:
        performance_benchmark.compute_time()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str, default='gogamza/kobart-summarization', help='model checkpoint (default: gogamza/kobart-summarization)')
    parser.add_argument('--test_dataset', type=str, default='metamong1/summarization_paper', help='test dataset (default: metamong1/summarization_paper)')
    parser.add_argument('--test_dataset_size', type=int, default=100, help='test dataset size (defualt: 100)')
    parser.add_argument('--cpu_flag', action='store_true', help='use cpu (default: gpu)')
    parser.add_argument('--test_categories', type=str, default='rouge,time,size', help='test categories seperated by , ex: time,size,rouge (defualt: rouge,time,size)')

    args = parser.parse_args()

    main(args)