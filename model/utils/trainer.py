import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Seq2SeqTrainer

from typing import Any, Dict, List, Optional, Tuple, Union
from packaging import version
from torch.optim.lr_scheduler import LambdaLR
from transformers.deepspeed import is_deepspeed_zero3_enabled

from copy import deepcopy

if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast
    
class Seq2SeqTrainerWithConditionalDocType(Seq2SeqTrainer):
    """
    original Trainer source code: https://github.com/huggingface/transformers/blob/master/src/transformers/trainer.py
    """
    def __init__(self, **kwargs):
        super(Seq2SeqTrainerWithConditionalDocType, self).__init__(**kwargs)
        
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

        gen_inputs = deepcopy(inputs)
        gen_inputs.pop("labels")

        generated_tokens = self.model.generate(
            **gen_inputs,
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
    
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.create_optimizer()
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)
    
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        if not self.args.is_noam:
            super().create_scheduler(num_training_steps, optimizer)
        else:
            if self.lr_scheduler is None:
                self.lr_scheduler = self.get_noam_schedule_with_warmup(
                    optimizer=self.optimizer if optimizer is None else optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                )
            return self.lr_scheduler

    def get_noam_schedule_with_warmup(self, optimizer, num_warmup_steps, last_epoch=-1):
        def lr_lambda(current_step: int):
            return 1 / math.sqrt(self.args.model_config.d_model) * min(1/math.sqrt(current_step+1), (current_step+1) /(num_warmup_steps**(1.5)))
        return LambdaLR(optimizer, lr_lambda, last_epoch)
        
    def get_normalized_probs(self, net_output, log_probs=True):
        """
        Get network output(loss, logits) and normalize logits using softmax
        Args:
            net_output tuple(loss, logits): logits before softmax
            log_probs bool: whether it is log probabilities
        Return:
            normalized probs: after softmax
        """
        logits = net_output["logits"] if isinstance(net_output, dict) else net_output[0]
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """[]
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        if not self.args.use_rdrop:
            return super().training_step(model, inputs)
            
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        concat_inputs = {
            'input_ids': torch.cat([inputs['input_ids'], inputs['input_ids'].clone()], 0),
            'attention_mask': torch.cat([inputs['attention_mask'], inputs['attention_mask'].clone()], 0),
            'labels': torch.cat([inputs['labels'], inputs['labels'].clone()], 0),
            # 'decoder_input_ids': torch.cat([inputs['decoder_input_ids'], inputs['decoder_input_ids'].clone()], 0),
        } # 두 번 forward 하기 힘드니까 concate해서 한 번에 feed 하고 잘라주는 형식입니다.

        if 'doc_type_ids' in inputs:
            concat_inputs['doc_type_ids'] = torch.cat([inputs['doc_type_ids'], inputs['doc_type_ids'].clone()], 0)\
                
        if self.use_amp:
            if version.parse(torch.__version__) >= version.parse("1.10"):
                with autocast(dtype=self.amp_dtype):
                    loss = self.compute_loss(model, concat_inputs)
            else:
                with autocast():
                    loss = self.compute_loss(model, concat_inputs)
        else:
            loss = self.compute_loss(model, concat_inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    # huggingface Trainer compute_loss 함수를 수정하였습니다.
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        
        if not self.args.use_rdrop and self.args.label_smoothing_factor == 0:
            return super().compute_loss(model, inputs)

        elif not self.args.use_rdrop and self.args.label_smoothing_factor != 0:
            assert "labels" in inputs
            labels = inputs["labels"]
            outputs = model(**inputs)
            # Save past state if it exists
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            if labels is not None:
                loss = self.label_smoother(outputs, labels)
            else:
                # We don't use .loss here since the model may return tuples instead of ModelOutput.
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            return (loss, outputs) if return_outputs else loss

        else:
            # if self.label_smoother is not None and "labels" in inputs:
            if "labels" in inputs:
                labels = inputs['labels'] # inputs.pop("labels")
                pad_mask = labels.unsqueeze(-1).eq(-100) # ignore_index
                # pad_mask = labels.unsqueeze(-1).eq(self.label_smoother.ignore_index)
                # labels = torch.cat([labels, labels.clone()], 0) # for r-drop3
            else:
                labels = None
            
            outputs = model(**inputs)
            
            # Save past state if it exists
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            if labels is not None:
                # loss = self.label_smoother(outputs, labels)
                loss = self.label_smoothed_nll_loss(outputs, labels,
                                                    epsilon=0.1 if self.label_smoother else 0)
                kl_loss = self.compute_kl_loss(outputs, pad_mask=pad_mask)
                loss += self.args.reg_alpha * kl_loss
            else:
                # We don't use .loss here since the model may return tuples instead of ModelOutput.
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            return (loss, outputs) if return_outputs else loss

    def compute_kl_loss(self, net_output, pad_mask=None, reduce=True):
        net_prob = self.get_normalized_probs(net_output, log_probs=True)
        net_prob_tec = self.get_normalized_probs(net_output, log_probs=False)

        p, q = torch.split(net_prob, net_prob.size(0)//2, dim=0)
        p_tec, q_tec = torch.split(net_prob_tec, net_prob_tec.size(0)//2, dim=0)
        
        p_loss = F.kl_div(p, q_tec, reduction='none')
        q_loss = F.kl_div(q, p_tec, reduction='none')
        
        if pad_mask is not None:
            pad_mask, _ = torch.split(pad_mask, pad_mask.size(0)//2, dim=0)
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)

        if reduce:
            p_loss = p_loss.sum()
            q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss

    def label_smoothed_nll_loss(self, model_output, labels, epsilon):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        log_probs = -F.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(-100) # self.label_smoother.ignore_index
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        nll_loss = nll_loss.sum()
        smoothed_loss = smoothed_loss.sum()
        eps_i = epsilon / log_probs.size(-1)
        return (1. - epsilon) * nll_loss + eps_i * smoothed_loss