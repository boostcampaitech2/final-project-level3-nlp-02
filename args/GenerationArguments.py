from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GenerationArguments:
    """
    Arguments generate model to summarization.
    https://huggingface.co/transformers/main_classes/model.html
    """
    max_length: Optional[int] = field(
        default=None,
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
        default=3,
        metadata={"help": "If set to int > 0, all ngrams of that size can only occur once."},
    )
    num_return_sequences: Optional[int] = field(
        default=1,
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
