from transformers.data.data_collator import DataCollatorForSeq2Seq

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