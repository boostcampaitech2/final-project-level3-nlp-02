

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from transformers.file_utils import PaddingStrategy
from transformers.models.bert import BertTokenizer, BertTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorForLanguageModeling

InputDataClass = NewType("InputDataClass", Any)

"""
A DataCollator is a function that takes a list of samples from a Dataset and collate them into a batch, as a dictionary
of PyTorch tensors.
"""

def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    import numpy as np
    import torch

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.
    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result


def pad_to_window(window_size:int, input_ids_tensor:torch.Tensor, tokenizer:PreTrainedTokenizerBase) :
    """longformer self attention의 구조로 인해서 input에 들어가는 seq_size가 attention_window로 반드시 나누어져야 하기 때문에 padding하는 함수가 필요"""
    batch_size, seq_size = input_ids_tensor.shape
    pad_flag = False if seq_size % window_size == 0 else True

    if pad_flag == False :
        return input_ids_tensor
    
    window_num = seq_size // window_size
    target_size = window_size * (window_num + 1)
    pad_size = target_size - seq_size

    if tokenizer.padding_side == 'right' :
        input_ids_tensor = F.pad(input_ids_tensor, (0,pad_size), mode='constant', value=tokenizer.pad_token_id)
    else :
        input_ids_tensor = F.pad(input_ids_tensor, (pad_size,0), mode='constant', value=tokenizer.pad_token_id)
    return input_ids_tensor

@dataclass
class DataCollatorForInfilling(DataCollatorForLanguageModeling):
    tokenizer: PreTrainedTokenizerBase
    poisson: int = 3
    label_pad_token_id: int = -100
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    window_size: int = 128

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        examples = self.pad_doc_type_ids(examples)
        batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        batch["input_ids"], batch["labels"] = self.torch_infilling(batch["input_ids"], poisson_num=self.poisson) 
        batch["input_ids"] = pad_to_window(self.window_size, batch["input_ids"], self.tokenizer)

        pad_size = batch['input_ids'].shape[1] - batch['attention_mask'].shape[1]
        if self.tokenizer.padding_side == 'right' :
            batch['attention_mask'] = F.pad(batch['attention_mask'], (0,pad_size), mode='constant', value=0)
            batch['doc_type_ids'] = F.pad(batch['doc_type_ids'], (0,pad_size))
        else :
            batch['attention_mask'] = F.pad(batch['attention_mask'], (pad_size,0), mode='constant', value=0)
            batch['doc_type_ids'] = F.pad(batch['doc_type_ids'], (pad_size,0))

        # label에서 pad_token_id로 된 부분은 loss 계산에서 제외하기 위해서 label_pad_token_id로 변경한다.
        batch["labels"] = torch.where(batch["labels"] == self.tokenizer.pad_token_id, self.label_pad_token_id, batch["labels"])
        return batch

    def pad_doc_type_ids(self, features) :
        doc_type_ids = [feature["doc_type_ids"] for feature in features] if "doc_type_ids" in features[0].keys() else None
        if doc_type_ids is not None:
            max_label_length = max(len(l) for l in doc_type_ids)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [0] * (max_label_length - len(feature["doc_type_ids"]))
                if isinstance(feature["doc_type_ids"], list):
                    feature["doc_type_ids"] = (
                        feature["doc_type_ids"] + remainder if padding_side == "right" else remainder + feature["doc_type_ids"]
                    )
                elif padding_side == "right":
                    feature["doc_type_ids"] = np.concatenate([feature["doc_type_ids"], remainder]).astype(np.int64)
                else:
                    feature["doc_type_ids"] = np.concatenate([remainder, feature["doc_type_ids"]]).astype(np.int64)
        return features


    def torch_infilling(self, inputs: Any, poisson_num: Optional[int] = None) -> Tuple[Any, Any]:
        import torch
        labels = inputs.clone() # label은 기존 문장과 동일
        pad_token_id = self.tokenizer.pad_token_id
        mask_token_id = self.tokenizer.mask_token_id
        batch_size, seq_size = inputs.shape

        poisson_value_list = np.random.poisson(poisson_num, batch_size) # batch size에 맞게 possion 값 생성
        input_list = []
        max_size = 0
        for i, poisson in enumerate(poisson_value_list) :
            input_arr = list(inputs[i])
            poisson = poisson_value_list[i] # 해당 문장에 대한 poisson 값
            pad_ids = np.where(np.array(input_arr) == pad_token_id)[0] 
            pad_size = len(pad_ids) # pad의 길이를 파악

            sen_size = seq_size - pad_size # sequence size - padding size : 기존의 input size

            if sen_size < poisson : # poisson으로 구한 값이 문장의 길이보다 긴 경우 
                max_size = max(max_size, len(input_arr))
                input_list.append(input_arr)
                continue

            infilling_start = np.random.randint(sen_size-poisson) # poisson value의 길이에 맞는 span의 시작점 선정
            infilling_end = infilling_start + poisson

            input_arr = input_arr[:infilling_start] + [mask_token_id] + input_arr[infilling_end:] # poisson value에 해당되는 길이의 span 자체를 mask 처리
            max_size = max(max_size, len(input_arr)) # 문장의 길이 파악
            input_list.append(input_arr)

        input_infilling = []
        # 각각의 문장의 길이가 다르기 때문에 Batch 중에서 가장 긴 길이에 맞춰서 padding을 진행한다.
        for input_ids in input_list :
            if self.tokenizer.padding_side == 'right' :
                input_ids = input_ids + [pad_token_id] * (max_size - len(input_ids))
            else :
                input_ids = [pad_token_id] * (max_size - len(input_ids)) + input_ids
            input_infilling.append(torch.tensor(input_ids))
            
        # list로 된 input list를 torch tensor로 변환해준다.
        input_infilling = torch.stack(input_infilling, dim=0)
        return input_infilling, labels
