import datasets
from dotenv import load_dotenv
import transformers
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
import torch

import os
import argparse

from .performanceBenchmark import PerformanceBenchmark


load_dotenv(verbose=True)

def performance_test(
    *,
    check_point = 'gogamza/kobart-summarization',
    test_dataset = 'metamong1/summarization',
    test_dataset_size = 1000,
    cpu_flag=False,
    test_categories='rouge,time,size',
    tokenizer=None,
    model=None,
    seed=42,
    args=None
):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    api_token = os.getenv('USE_AUTH_TOKEN')
    if args:
        check_point = args.check_point
        test_dataset = args.test_dataset
        test_dataset_size = args.test_dataset_size
        cpu_flag = args.cpu_flag
        test_categories = args.test_categories

    if cpu_flag:
        device='cpu'

    dataset = datasets.load_dataset(test_dataset, use_auth_token=api_token)
    test_dataset = dataset['validation'].shuffle(seed=seed).filter(lambda x: len(x['text'])< 500).select(range(test_dataset_size))
    
    if not tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(check_point)

    
    if not model:
        model = AutoModelForSeq2SeqLM.from_pretrained(check_point, torch_dtype='auto')
    
    model = model.to(device)

    summerizer = pipeline(
        'summarization', 
        model=model,
        tokenizer=tokenizer,
        device = 0 if torch.cuda.is_available() and not cpu_flag else -1
    )

    performance_benchmark = PerformanceBenchmark(summerizer, test_dataset, tokenizer, 'baseline')

    test_categories = test_categories.split(',')
    if 'rouge' in test_categories:
        performance_benchmark.compute_rouge()
    
    if 'size' in test_categories:
        performance_benchmark.compute_size()
    
    if 'time' in test_categories:
        performance_benchmark.compute_time()

def main(args):
    performance_test(args=args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--check_point', type=str, default='gogamza/kobart-summarization', help='model checkpoint (default: gogamza/kobart-summarization)')
    parser.add_argument('--test_dataset', type=str, default='metamong1/summarization', help='test dataset (default: metamong1/summarization)')
    parser.add_argument('--test_dataset_size', type=int, default=1000, help='test dataset size (defualt: 1000)')
    parser.add_argument('--cpu_flag', action='store_true', help='use cpu (default: gpu)')
    parser.add_argument('--test_categories', type=str, default='rouge,time,size', help='test categories seperated by , ex: time,size,rouge (defualt: rouge,time,size)')

    args = parser.parse_args()

    main(args)