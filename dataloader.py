from typing import List
from torch.utils.data import Dataset
from preprocessor import DocsPreprocessor
from datasets import (
    load_dataset,
    concatenate_datasets
)

class SumDataset(Dataset) :
    """ 
    Summarization 여러 데이터셋을 merge하여 사용하기 위한 class.
    Args:
        data_types (List[str]): 업로드한 huggingface datasets 중, 원하는 datasets 이름의 list
        mode (str): train or validation 선택하는 변수
    """
    def __init__(self,
        data_types: List[str],
        mode: str,
        USE_AUTH_TOKEN: str
    ) :
        self.dataset = []
        self.data_preprocessor = DocsPreprocessor()
        self.mode=mode

        for data_type in data_types :          
            dataset_idx = load_dataset(data_type, use_auth_token=USE_AUTH_TOKEN)

            if mode == 'test' :
                dataset_idx = dataset_idx.map(self.data_preprocessor.for_test)
            else :
                dataset_idx = dataset_idx.map(self.data_preprocessor.for_train)

            dataset_idx.cleanup_cache_files() # 전처리 성능 실험을 위해서 cache 지우는 과정
            self.dataset.append(dataset_idx)

    def load_data(self):
        dataset = concatenate_datasets([ds[self.mode] for ds in self.dataset])
        return dataset

    def __len__(self):
        return len(self.dataset)