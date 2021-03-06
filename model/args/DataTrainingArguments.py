from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    text_column: Optional[str] = field(
        default='text',
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default='title',
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=1,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: int = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    use_auth_token_path: Optional[str] = field(
        default='./use_auth_token.env',
        metadata={"help":'input your use_auth_token path'},
    )
    num_samples: int = field(
        default=None,
        metadata={  
            "help": "Set number of data sampling"
        },
    )
    relative_eval_steps: int = field(
        default=None,
        metadata={  
            "help": "Calculate the evaluation step relative to the size of the data set."
        },
    )
    is_pretrain : bool = field(
        default=False,
        metadata={
            "help" : "Whether to pretrain model with infilling masking task"
        }
    )
    use_doc_type_ids: bool = field(
        default=False,
        metadata={  
            "help": "Calculate the evaluation step relative to the size of the data set."
        },
    )
    is_part: bool = field(
         default=False, 
         metadata={ 
            "help": "whether to ba a part of datasets (default=False)" 
        }, 
    )
    compute_filter_stopwords: bool = field(
         default=False, 
         metadata={ 
            "help": "whether to ba a part of datasets (default=False)" 
        }, 
    )
    use_preprocessing: bool = field(
         default=False, 
         metadata={ 
            "help": "whether to preprocess(default=False)" 
        }, 
    )
    is_valid: bool = field(
         default=False, 
         metadata={ 
            "help": "is validation datasets" 
        }, 
    )