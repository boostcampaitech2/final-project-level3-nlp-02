import torch
from typing import List
from torch.utils.data import Dataset


from datasets import load_dataset, concatenate_datasets


class SumDataset(Dataset) :
    def __init__(self,
        data_types: List[str],
        mode: str
    ) :
        self.dataset = []
        self.mode=mode
        for data_type in data_types :
            self.dataset.append(load_dataset(data_type, use_auth_token=True))
    def load_data(self):
        dataset = concatenate_datasets([ds[self.mode] for ds in self.dataset])
        return dataset

    def __len__(self):
        return len(self.dataset)