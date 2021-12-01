from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="klue/roberta-large",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default="klue/roberta-large",
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

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    learning_rate: Optional[float] = field(
        default=5e-5,
        metadata = {"help":"The initial learning rate for Adam."}
    )

    num_train_epochs: Optional[float] = field(
        default=3.0,
        metadata = {"help":"Total number of training epochs to perform."}
    )

    warmup_ratio: Optional[float] = field(
        default=0.1,
        metadata = {"help" : "Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training."}
    )

    gradient_accumulation_steps: Optional[int] = field(
        default = 1,
        metadata = {"help":"Number of updates steps to accumulate before performing a backward/update pass."}
    )

    dataset_name: Optional[str] = field(
        default="metamong1/summarization_paper",
        metadata={"help": "The name of the dataset to use."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=2,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True, #True
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    eval_retrieval: str = field(
        default="sparse",
        metadata={
            "help": "Choose which passage retrieval to be used.[sparse, elastic_sparse]."
        },
    )
    num_clusters: int = field(
        default=64, metadata={"help": "Define how many clusters to use for faiss."}
    )
    top_k_retrieval: int = field(
        default=50,
        metadata={
            "help": "Define how many top-k passages to retrieve based on similarity."
        },
    )
    score_ratio: float = field(
        default=0,
        metadata={
            "help": "Define the score ratio."
        },
    )
    train_retrieval: bool = field(
        default=False,
        metadata={"help": "Whether to train sparse/dense embedding (prepare for retrieval)."},
    )
    data_selected: str = field(
        default="",
        metadata={"help": "data to find added tokens, context/answers/question with '_' e.g.) context_answers"},
    )
    rtt_dataset_name:str = field(
        default=None,
        metadata={"help" : "input rtt data name with path"},
    )
    preprocessing_pattern:str = field(
        default=None,
        metadata={"help" : "preprocessing(e.g. 123)"},
    )
    add_special_tokens_flag:bool = field(
        default=False,
        metadata={"help": "add special tokens"},
    )
    add_special_tokens_query_flag:bool = field(
        default=False,
        metadata={"help": "add special tokens about question type"},
    )
    retrieve_pickle: str = field(
        default='',
        metadata={"help":"put a pickle file path for load"},
    )
    another_scheduler_flag :bool = field(
        default=False,
        metadata={"help": "create another scheduler"}
    )
    num_cycles :int = field(
        default=1,
        metadata={"help": "cycles for get_cosine_schedule_with_warmup"}
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
        default="mrc_project_1",
        metadata={"help": "project name"},
    )