import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

import argparse

import performance_test

def dynamic_quantization(
    *,
    check_point='gogamza/kobart-summarization',
    test_dataset = 'metamong1/summarization_paper',
    test_dataset_size = 1000,
    test_categories='rouge,time,size',
    tokenizer=None,
    model = None,
    test=True,
):
    if model:
        model_quantized = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    elif check_point:
        model = AutoModelForSeq2SeqLM.from_pretrained(check_point)
        model_quantized = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

    if test:
        performance_test.performance_test(
            check_point=check_point,
            test_dataset=test_dataset,
            test_dataset_size=test_dataset_size,
            cpu_flag=True,
            test_categories=test_categories,
            tokenizer=tokenizer,
            model=model_quantized, 
        )
    
    return model_quantized


def half_quantization(
    *,
    check_point='gogamza/kobart-summarization',
    test_dataset = 'metamong1/summarization_paper',
    test_dataset_size = 1000,
    test_categories='rouge,time,size',
    tokenizer=None,
    model = None,
    test=True
):
    
    if model:
        model.half()
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()
    elif check_point:
        model = AutoModelForSeq2SeqLM.from_pretrained(check_point)
        model = model.half()
        
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()
    
    if test:
        performance_test.performance_test(
            check_point=check_point,
            test_dataset=test_dataset,
            test_dataset_size=test_dataset_size,
            cpu_flag=False,
            test_categories=test_categories,
            tokenizer=tokenizer,
            model=model,
        )
    
    return model


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.check_point)

    if args.quantization_type == 'half_quantization':
        model = half_quantization(
            check_point=args.check_point,
            test_dataset=args.test_dataset,
            test_dataset_size=args.test_dataset_size,
            test_categories=args.test_categories,
            tokenizer=tokenizer,
            test= args.no_test_flag,
        )
    elif args.quantization_type == 'dynamic_quantization':
        model = dynamic_quantization(
            check_point=args.check_point,
            test_dataset=args.test_dataset,
            test_dataset_size=args.test_dataset_size,
            test_categories=args.test_categories,
            tokenizer=tokenizer,
            test= args.no_test_flag,
        )
    
    if args.save_dir:
        model.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)
        torch.save(model.state_dict(), args.save_dir+'.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--quantization_type', type=str, default='half_quantization', help='quantization type. ex: half_quantization, dynamic_quantization (default: half_quantization)')
    parser.add_argument('--check_point', type=str, default='gogamza/kobart-summarization', help='model checkpoint (default: gogamza/kobart-summarization)')
    parser.add_argument('--test_dataset', type=str, default='metamong1/summarization_paper', help='test dataset (default: metamong1/summarization_paper)')
    parser.add_argument('--test_dataset_size', type=int, default=1000, help='test dataset size (defualt: 1000)')
    parser.add_argument('--cpu_flag', action='store_true', help='use cpu (default: gpu)')
    parser.add_argument('--test_categories', type=str, default='rouge,time,size', help='test categories seperated by , ex: time,size,rouge (defualt: rouge,time,size)')
    parser.add_argument('--no_test_flag', action='store_false', help='do test performance (default: False)')
    parser.add_argument('--save_dir', type=str, default='', help='save model directory (default: "" ')

    args = parser.parse_args()

    main(args)