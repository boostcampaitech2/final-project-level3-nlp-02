from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments :
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="gogamza/kobart-base-v1",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
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
    attention_window_size: int = field(
        default=256,
        metadata={
            "help": "attention window size"
        },
    )
    longformerbart_path: str = field(
        default='./model/longformerbart',
        metadata={
            "help": "path for loading longformerbart model"
        },
    )
    hidden_size: int = field(
        default=128,
        metadata={
            "help": "hidden dimension size / h_model size"
        },
    )
    encoder_layer_size: int = field(
        default=3,
        metadata={
            "help": "number of encoder layers"
        },
    )
    decoder_layer_size: int = field(
        default=3,
        metadata={
            "help": "number of decoder layers"
        },
    )
    attention_head_size: int = field(
        default=4,
        metadata={"help": "number of attention heads"},
    )
    attention_window_size: int = field(
        default=256,
        metadata={
            "help": "attention window size"
        },
    )
    dropout: float = field(
        default=0.1,
        metadata={
            "help":"dropout ratio"
        },
    )
    use_model: str = field(
        default='auto',
        metadata={"help": "model type(pretrained model from huggingface, customized bigbart, customized longbart), [auto, bigbart, longbart, bigbart_tapt]"},
    )
    
