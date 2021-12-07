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
        shuffle_seed: int,
        ratio: float,
        USE_AUTH_TOKEN: str,
    ) :
        self.dataset = []
        self.data_preprocessor = DocsPreprocessor()
        self.mode=mode
        self.ratio = ratio
        self.shuffle_seed = shuffle_seed
        for data_type in data_types :
            self.dataset.append(load_dataset(data_type, use_auth_token=USE_AUTH_TOKEN))
    
    def load_data(self):
        dataset_list = []
        for ds in self.dataset :
            typed_ds = ds[self.mode]
            sampling_count = round(len(typed_ds)*self.ratio)
            
            if self.mode == 'test' :
                typed_ds = typed_ds.map(self.data_preprocessor.for_test)
            else :
                typed_ds = typed_ds.map(self.data_preprocessor.for_train)

            sampled_data_ds = typed_ds.shuffle(self.shuffle_seed).select(range(sampling_count))
            dataset_list.append(sampled_data_ds)
        dataset = concatenate_datasets(dataset_list)
        return dataset

    def __len__(self):
        return len(self.dataset)