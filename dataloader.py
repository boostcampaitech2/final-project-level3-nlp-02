from typing import List
from torch.utils.data import Dataset
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
        self.mode=mode
        for data_type in data_types :
            self.dataset.append(load_dataset(data_type, use_auth_token=USE_AUTH_TOKEN))
    def load_data(self):
        dataset = concatenate_datasets([ds[self.mode] for ds in self.dataset])
        return dataset

    def __len__(self):
        return len(self.dataset)