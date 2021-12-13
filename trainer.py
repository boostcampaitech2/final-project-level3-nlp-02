import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Seq2SeqTrainer

from typing import Any, Dict, List, Optional, Tuple, Union
from packaging import version
from torch.optim.lr_scheduler import LambdaLR
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer import Trainer
from transformers.trainer_utils import PredictionOutput
from transformers.utils import logging

from transformers.trainer import number_of_arguments


if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_torch_generator_available = True
    _is_native_amp_available = True
    from torch.cuda.amp import autocast


class Seq2SeqTrainerWithDocType(Seq2SeqTrainer):
    """
    original Trainer source code: https://github.com/huggingface/transformers/blob/master/src/transformers/trainer.py
    """
    def __init__(self, **kwargs):
        super(Seq2SeqTrainerWithDocType, self).__init__(**kwargs)
        
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
        }

        generated_tokens = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            doc_type_ids=inputs["doc_type_ids"],
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return (loss, generated_tokens, labels)

    def call_model_init(self, trial=None):
        model_init_argcount = number_of_arguments(self.model_init)
        if model_init_argcount == 1:
            model = self.model_init(self.args)
        elif model_init_argcount == 2:
            model = self.model_init(self.args, trial)
        else:
            raise RuntimeError("model_init should have 0 or 1 argument.")
        if model is None:
            raise RuntimeError("model_init should not return None.")
        return model
    
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.create_optimizer()
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)
    
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        if not self.args.is_noam:
            super().create_scheduler(num_training_steps, optimizer)
        else:
            if self.lr_scheduler is None:
                self.lr_scheduler = self.get_noam_schedule_with_warmup(
                    #self.args.lr_scheduler_type,
                    optimizer=self.optimizer if optimizer is None else optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    #num_training_steps=num_training_steps,
                )
            return self.lr_scheduler

    def get_noam_schedule_with_warmup(self, optimizer, num_warmup_steps, last_epoch=-1):
        def lr_lambda(current_step: int):
            return 1 / math.sqrt(self.args.model_config.d_model) * min(1/math.sqrt(current_step+1), (current_step+1) /(num_warmup_steps**(1.5)))
        return LambdaLR(optimizer, lr_lambda, last_epoch)