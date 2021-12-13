from dataclasses import dataclass, field
from typing import Optional
from transformers  import Seq2SeqTrainingArguments

@dataclass
class CustomSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    """
    Early stopping 및 파일 scheduling을 위한 arguments 선정
    """
    save_strategy : str = field(
        default="steps",
        metadata={"help": "save strategy to adopt during training"
        }
    )
    save_steps : int = field(
        default=500,
        metadata={"help": "Number of updates steps before two checkpoint saves"
        }
    )
    save_total_limit: int = field(
        default=5,
        metadata={"help": "If a value is passed, will limit the total amount of checkpoints."
        "Deletes the older checkpoints in :obj:`output_dir`."}
    )
    metric_for_best_model: str = field(
        default="loss",
        metadata={
            "help": "to specify the metric to use to compare two different models"
            "*loss, *rouge1, *rouge2, *rougeL, *rougeLsum"}
    )
    greater_is_better: Optional[bool] = field(
        default=None, metadata={"help": "Whether the `metric_for_best_model` should be maximized or not."}
    )
    eval_steps: Optional[int] = field(
        default=500,
        metadata={
            "help" : "Number of update steps between two evaluations"
        },
    )
    evaluation_strategy: str = field(
        default="steps",
        metadata={
            "help": "The evaluation strategy to adopt during training"
        },
    )
    load_best_model_at_end: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to load the best model found during training at the end of training"
        },
    )
    logging_steps: int = field(
        default=1000,
        metadata={  
            "help": "Number of update steps between two logs"
        },
    )
    es_patience: int = field(
        default=5,
        metadata={  
            "help": "patience steps for early stopping"
        },
    )
    alpha: float = field(
        default=0.5,
        metadata={
            "help": "alpha value for distillation methods(default: 0.5)"
        }
    )
    temperature: float = field(
        default=1.0,
        metadata={
            "help": "temperature value for distillation methods(default: 1.0)"
        }
    )
    use_original: bool = field(
        default=False,
        metadata={
            "help": "whether to use original prediction loss as paper for tiny distillation(default: False)"
        }
    )
    distillation_type: str = field(
        default=None,
        metadata={
            "help": "which distillation method to use (ex. distil, tiny, None) (defualt: None)"
        }
    )