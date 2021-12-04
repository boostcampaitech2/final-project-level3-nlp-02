from dataclasses import dataclass, field
from typing import Optional
from transformers  import Seq2SeqTrainingArguments

@dataclass
class CustomSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    """
    Early stopping 및 파일 scheduling을 위한 arguments 선정
    """
    save_total_limit: int = field(
        default=6,
        metadata={"help": "If a value is passed, will limit the total amount of checkpoints."
        "Deletes the older checkpoints in :obj:`output_dir`."}
    )
    metric_for_best_model: str = field(
        default="rouge_l",
        metadata={
            "help": "to specify the metric to use to compare two different models"}
    )
    eval_steps: Optional[int] = field(
        default=500,
        metadata={
            "help": "The `max_length` to use on each evaluation loop when `predict_with_generate=True`. Will default "
            "to the `max_length` value of the model configuration."
        },
    )
    evaluation_strategy: str = field(
        default="steps",
        metadata={
            "help": "The evaluation strategy to adopt during training"
        },
    )
    load_best_model_at_end: bool = field(
        default=True,
        metadata={
            "help": ""
        },
    )
    logging_steps: int = field(
        default=1000,
        metadata={
            "help": ""
        },
    )