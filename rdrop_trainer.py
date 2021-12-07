import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Seq2SeqTrainer

from typing import Any, Dict, List, Optional, Tuple, Union
from packaging import version

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_torch_generator_available = True
    _is_native_amp_available = True
    from torch.cuda.amp import autocast


class RdropTrainer(Seq2SeqTrainer):
    """
    original Trainer source code: https://github.com/huggingface/transformers/blob/master/src/transformers/trainer.py
    """
    def __init__(self, **kwargs):
        super(RdropTrainer, self).__init__(**kwargs)

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
        """
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
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        concat_inputs = {
            'input_ids': torch.cat([inputs['input_ids'], inputs['input_ids'].clone()], 0),
            'attention_mask': torch.cat([inputs['attention_mask'], inputs['attention_mask'].clone()], 0),
            'labels': torch.cat([inputs['labels'], inputs['labels'].clone()], 0),
            'decoder_input_ids': torch.cat([inputs['decoder_input_ids'], inputs['decoder_input_ids'].clone()], 0),
        } # 두 번 forward 하기 힘드니까 concate해서 한 번에 feed 하고 잘라주는 형식입니다.

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

        # if self.do_grad_scaling: # transformers==4.11.0 에는 없어서 주석처리 하였습니다.
        #     self.scaler.scale(loss).backward()
        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
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
        
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
            pad_mask = labels.unsqueeze(-1).eq(self.label_smoother.ignore_index)
            # labels = torch.cat([labels, labels.clone()], 0) # for r-drop
        else:
            labels = None
        outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            # loss = self.label_smoother(outputs, labels)
            loss = self.label_smoothed_nll_loss(outputs, labels, 0.1)
            kl_loss = self.compute_kl_loss(outputs, pad_mask)
            loss += 0.7 * kl_loss # 0.7 == reg_alpha
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

        padding_mask = labels.eq(self.label_smoother.ignore_index)
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