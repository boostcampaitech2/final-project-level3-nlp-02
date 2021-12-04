from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LoggingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dotenv_path: Optional[str] = field(
        default='./wandb.env',
        metadata={"help":'input your dotenv path'},
    )
    wandb_name: Optional[str] = field(
        default="model/kobart",
        metadata={"help": "wandb name"},
    )
    wandb_unique_tag: Optional[str] = field(
        default=None,
        metadata={"help":"input your wandb unique tag"},
    )
    project_name: Optional[str] = field(
        default="kobart",
        metadata={"help": "project name"},
    )
    dotenv_path: Optional[str] = field(
        default="/opt/ml/final-project-level3-nlp-02/wandb.env",
        metadata={"help":"input your dotenv path"},
    )