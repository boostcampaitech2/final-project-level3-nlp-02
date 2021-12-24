import sys
sys.path.append('..')


import numpy as np
import torch

from pathlib import Path
from time import perf_counter

from utils.rouge import compute


class PerformanceBenchmark:
    def __init__(self, pipeline, dataset, tokenizer, optim_type='base line'):
        self.pipeline = pipeline
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.optim_type = optim_type

    def compute_rouge(self):
        rouge_scores = {}
        pred = self.pipeline(self.dataset['text'])
        label = self.dataset['title']

        pred = [key['summary_text'] for key in pred]
        rouge_score = compute(pred, label, self.tokenizer)

        for key, value in rouge_score.items():
            if key not in rouge_scores:
                rouge_scores[key] = round(rouge_score[key].mid.fmeasure * 100, 4)
        
        print("====ROUGE score====")
        print(rouge_scores)
        return rouge_scores

    def compute_size(self):
        state_dict = self.pipeline.model.state_dict()
        path = Path("model.pt")
        torch.save(state_dict, path)

        size_mb = Path(path).stat().st_size / (1024 * 1024)

        path.unlink()

        print(f"Model size (MB) = {size_mb}")
        return {'size_mb': size_mb}

    def compute_time(self, query :str = '최근 대부분의 범죄에 디지털 매체가 사용되면서 디지털 데이터는 필수 조사 대상이 되었다. 하지만 디지털 데이터는 비교적 쉽게 삭제 및변조가 가능하다. 따라서 디지털 증거 획득을 위해 삭제된 데이터의 복구가 필요하며, 파일 카빙은 컴퓨터 포렌식 조사에서 증거를 획득할 수있는 중요한 요소이다. 하지만 현재 사용되는 파일 카빙 도구들은 포렌식 조사를 위한 데이터의 선별을 고려하지 않고 있다. 또 기존의 파일카빙 기법들은 파일의 일부 영역이 덮어써지거나 조각날 경우 복구가 불가능한 단점이 있다. 따라서 본 논문에서는 포렌식 조사시 유용한 정보를 획득할 수 있는 파일을 제안하고, 기존의 파일 카빙 기법보다 효과적으로 데이터를 복구할 수 있는 레코드 파일 카빙 기법을 제시한다.'):
        times = []

        for i in range(10):
            self.pipeline(query)

        for i in range(100):
            start_time = perf_counter()
            self.pipeline(query)
            time = perf_counter() - start_time
            times.append(time)

        time_avg_ms = 1000 * np.mean(times)
        time_std_ms = 1000 * np.std(times)

        print(f"Average time took(ms) {time_avg_ms:.2} +\- {time_std_ms:.2f}")
        return {"time_avg_ms": time_avg_ms, "time_std_ms": time_std_ms}


    def run_benchmark(self):
        metrics = {}
        metrics[self.optim_type] = self.compute_size()
        metrics[self.optim_type].update(self.compute_time())
        metrics[self.optim_type].update(self.compute_accuracy())
        return metrics