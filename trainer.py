
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.trainer_seq2seq import Seq2SeqTrainer
from typing import Any, Dict, List, Optional, Tuple, Union
from packaging import version

from transformers.optimization import (
    AdamW,
    get_cosine_with_hard_restarts_schedule_with_warmup
)

from transformers.deepspeed import is_deepspeed_zero3_enabled

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_torch_generator_available = True
    _is_native_amp_available = True
    from torch.cuda.amp import autocast


class Seq2SeqTrainerWithConsineWithReStartScheduler(Seq2SeqTrainer):
    """
    original Trainer source code: https://github.com/huggingface/transformers/blob/master/src/transformers/trainer.py
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_optimizer_and_scheduler(self, num_training_steps: int, num_cycles:int = 1, another_scheduler_flag=False):
        if not another_scheduler_flag:
            self.create_optimizer()
            self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)
        else:
            optimizer_kwargs = {
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                    "lr" : self.args.learning_rate,
                }

            self.optimizer = AdamW(self.model.parameters(), **optimizer_kwargs)
            self.lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                                self.optimizer, num_warmup_steps=self.args.warmup_steps, 
                                num_training_steps= num_training_steps,
                                num_cycles = num_cycles)