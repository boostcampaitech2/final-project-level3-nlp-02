from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="gogamza/kobart-summarization",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default="gogamza/kobart-summarization",
        metadata={
            "help": "Pretrained config name or path if not the s ame as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "customized tokenizer path if not the same as model_name"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default="paper,news",
        metadata={"help": "The name of the dataset to use."},
    )
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
        default=2,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: int = field(
        default=256,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=256,
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
        default=5,
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

@dataclass
class LoggingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    wandb_name: Optional[str] = field(
        default="model/roberta",
        metadata={"help": "wandb name"},
    )

    dotenv_path: Optional[str] = field(
        default='./wandb.env',
        metadata={"help":'input your dotenv path'},
    )

    project_name: Optional[str] = field(
        default="final_project",
        metadata={"help": "project name"},
    )

@dataclass
class GenArguments:
    """
    Arguments generate model to summarization.
    https://huggingface.co/transformers/main_classes/model.html
    """
    max_length: Optional[int] = field(
        default=50,
        metadata={"help": "maximum length of the sequence to be generated."},
    )
    min_length: Optional[int]  = field(
        default=2,
        metadata={"help": "minimum length of the sequence to be generated."},
    )
    length_penalty: Optional[float] = field(
        default=1,
        metadata={"help": "values < 1.0 in order to encourage the model to generate shorter sequences"},
    )
    early_stopping: bool = field(
        default=True,
        metadata={"help": "Whether to stop the beam search when at least num_beams sentences are finished per batch or not"},
    )
    output_scores: bool = field(
        default=True,
        metadata={"help": "Whether or not to return the prediction scores."},
    )
    no_repeat_ngram_size: Optional[int] = field(
        default=2,
        metadata={"help": "If set to int > 0, all ngrams of that size can only occur once."},
    )
    num_return_sequences: Optional[int] = field(
        default=5,
        metadata={"help": "The number of independently computed returned sequences for each element in the batch."},
    )
    top_k: Optional[int] = field(
        default=50,
        metadata={"help": "The number of highest probability vocabulary tokens to keep for top-k-filtering."},
    )
    top_p: Optional[float] = field(
        default=0.95,
        metadata={"help": "the most probable tokens with probabilities that add up to top_p or higher are kept for generation."},
    )
