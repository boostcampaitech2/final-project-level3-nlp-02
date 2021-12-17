from dataclasses import dataclass
import numpy as np
import torch
import math
from typing import Optional, Union, List, Any, Dict, Tuple
from transformers.data.data_collator import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
from transformers.tokenization_utils import PreTrainedTokenizerBase, BatchEncoding

class DataCollatorForSeq2SeqWithDocType(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        import numpy as np

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)


        doc_type_ids = [feature["doc_type_ids"] for feature in features] if "doc_type_ids" in features[0].keys() else None
        # We have to pad the doc_type_ids before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
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
        
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features

@dataclass
class DataCollatorForTextInfillingDocType:
    """
    Implementation of Text infilling 
    https://github.com/huggingface/transformers/pull/12370/files#

    """
    tokenizer: PreTrainedTokenizerBase
    label_pad_token_id: int = -100
    mlm_probability: float = 0.15
    poisson_lambda: float = 3.0
    pad_to_multiple_of: Optional[int] = None

    def __post_init__(self):
        if self.tokenizer.mask_token is None:
            raise ValueError

    def __call__(self, features: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
                 ) -> Dict[str, torch.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.
        doc_type_ids = [feature["doc_type_ids"] for feature in features] if "doc_type_ids" in features[0].keys() else None
        if doc_type_ids is not None:
            max_doc_type_length = max(len(l) for l in doc_type_ids)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                if  (max_doc_type_length % self.pad_to_multiple_of != 0) :
                    max_doc_type_length = ((max_doc_type_length // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

                remainder = [0] * (max_doc_type_length - len(feature["doc_type_ids"]))
                if isinstance(feature["doc_type_ids"], list):
                    feature["doc_type_ids"] = (
                        feature["doc_type_ids"] + remainder if padding_side == "right" else remainder + feature["doc_type_ids"]
                    )
                elif padding_side == "right":
                    feature["doc_type_ids"] = np.concatenate([feature["doc_type_ids"], remainder]).astype(np.int64)
                else:
                    feature["doc_type_ids"] = np.concatenate([remainder, feature["doc_type_ids"]]).astype(np.int64)


        if isinstance(features[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(features, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {"input_ids": self._torch_collate_batch(features, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)}
        
        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if doc_type_ids is not None:
            batch["input_ids"], batch["labels"], batch["doc_type_ids"] = self.mask_tokens(batch,special_tokens_mask)
        else :
            batch["input_ids"], batch["labels"] = self.mask_tokens(batch,special_tokens_mask)

        return batch

    def mask_tokens(self,
                    batch: Dict,
                    special_tokens_mask: Optional[torch.Tensor] = None,
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        doc_type_ids = batch["doc_type_ids"] if "doc_type_ids" in batch else None

        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        # determine how many tokens we need to mask in total
        is_token = ~(input_ids == self.tokenizer.pad_token_id) & ~special_tokens_mask
        num_to_mask = int(math.ceil(is_token.float().sum() * self.mlm_probability))

        if num_to_mask == 0:
            return input_ids, labels

        # generate a sufficient number of span lengths
        poisson_distribution = torch.distributions.Poisson(rate=self.poisson_lambda)
        lengths = poisson_distribution.sample(sample_shape=(num_to_mask,))
        while torch.cumsum(lengths, 0)[-1] < num_to_mask:
            lengths = torch.cat([lengths, poisson_distribution.sample(sample_shape=(num_to_mask,))])

        # remove all spans of length 0
        # Note that BART inserts additional mask tokens where length == 0,
        # which we do not implement for now as it adds additional complexity
        # lengths = lengths[lengths > 0]

        # trim to about num_to_mask tokens
        idx = torch.argmin(torch.abs(torch.cumsum(lengths, 0) - num_to_mask)) + 1
        lengths = lengths[:idx + 1]

        # select span start indices
        token_indices = is_token.nonzero(as_tuple=False)
        span_starts = torch.randperm(token_indices.shape[0])[:lengths.shape[0]]

        # prepare mask
        masked_indices = token_indices[span_starts]
        mask = torch.full_like(input_ids, fill_value=False)

        # mask span start indices
        for mi in masked_indices:
            mask[tuple(mi)] = True
        lengths -= 1

        # fill up spans
        max_index = input_ids.shape[1] - 1
        remaining = (lengths > 0) & (masked_indices[:, 1] < max_index)
        while torch.any(remaining):
            for mi in masked_indices:
                mask[tuple(mi)] = True
            lengths -= 1
            masked_indices[remaining, 1] += 1
            remaining = (lengths > 0) & (masked_indices[:, 1] < max_index)
        
        # place the mask tokens
        mask = mask.masked_fill_(special_tokens_mask, False)
        input_ids[mask.bool()] = self.tokenizer.mask_token_id
        label_mask = labels == self.tokenizer.pad_token_id
        labels[label_mask] = self.label_pad_token_id
    
        # remove mask tokens that are not starts of spans
        to_remove = mask.bool() & mask.bool().roll(shifts=(0,1), dims=(0,1))
        new_input_ids = torch.full_like(input_ids, fill_value=self.tokenizer.pad_token_id)

        new_doc_type_ids = torch.full_like(input_ids, fill_value=0) # check 2

        for i, example in enumerate(torch.split(input_ids, split_size_or_sections=1, dim=0)):
            new_example = example[0][~to_remove[i]]
            new_input_ids[i, 0:new_example.shape[0]] = new_example
            
            if doc_type_ids is not None :
                doc_type_ids_ = doc_type_ids[i][~to_remove[i]]
                new_doc_type_ids[i, 0:doc_type_ids_.shape[0]] = doc_type_ids_ 

        # return new_input_ids, labels, {"doc_type_ids" : doc_type_ids,"new_doc_type_ids" : new_doc_type_ids,"to_remove" : to_remove}
        if doc_type_ids is not None :
            return new_input_ids, labels, new_doc_type_ids
        else :
            return new_input_ids, labels
    
    def _torch_collate_batch(self, examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
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