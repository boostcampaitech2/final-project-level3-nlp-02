import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict,Union, Any
from torch.optim.lr_scheduler import LambdaLR
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from packaging import version
if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast

class DistillationTrainer(Seq2SeqTrainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
    
    def compute_loss(self, model, inputs, return_outputs=False):

        if self.args.label_smoothing_factor != 0:
            if "labels" in inputs:
                labels = inputs['labels']
                pad_mask = labels.unsqueeze(-1).eq(-100)

        outputs_student = model(**inputs)

        # Extract cross-entropy loss and logtis from student
        loss_cross_entropy = outputs_student.loss
        logits_student = outputs_student.logits
        

        # Extract logits from teacher
        with torch.no_grad():
            outputs_teacher = self.teacher_model(**inputs)
            logits_teacher = outputs_teacher.logits
        
        # Soften probabilities and compute distillation loss
        loss_fct = nn.KLDivLoss(reduction="batchmean")
        loss_kd = self.args.temperature ** 2 * loss_fct(
            F.log_softmax(logits_student / self.args.temperature, dim=-1),
            F.softmax(logits_teacher / self.args.temperature, dim=-1)
        )

        loss = self.args.alpha * loss_cross_entropy + (1. - self.args.alpha) * loss_kd
        
        if labels is not None:
            loss += self.label_smoothed_nll_loss(outputs_student, labels,
                                            epsilon=0.1 if self.label_smoother else 0)
    
        if self.args.use_rdrop:
            loss += self.args.reg_alpha * self.compute_kl_loss(outputs_student, pad_mask=pad_mask)

        # Return weighted student loss
        return (loss, outputs_student) if return_outputs else loss
    
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

class TinyTrainer(Seq2SeqTrainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
    
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
    
    def compute_loss(self, model, inputs, return_outputs=False):
        
        if self.args.label_smoothing_factor != 0:
            if "labels" in inputs:
                labels = inputs['labels']
                pad_mask = labels.unsqueeze(-1).eq(-100)

        # Student model output
        outputs_student = model(**inputs, output_hidden_states=True, output_attentions=True)
        # Teacher model output
        with torch.no_grad():
            outputs_teacher = self.teacher_model(**inputs, output_hidden_states=True, output_attentions=True)
        # Number of layers for each student and teacher models' encoder and decoder layers
        num_encoder_layers_student = len(outputs_student.encoder_attentions) 
        num_encoder_layers_teacher = len(outputs_teacher.encoder_attentions)
        num_decoder_layers_student = len(outputs_student.decoder_attentions)
        num_decoder_layers_teacher = len(outputs_teacher.decoder_attentions)

        # Layer ration between student and teacher model        
        encoder_layer_ratio = num_encoder_layers_teacher // num_encoder_layers_student
        decoder_layer_ratio = num_decoder_layers_teacher // num_decoder_layers_student

        # Define loss
        MSELoss = nn.MSELoss()

        # Loss initialization
        encoder_attention_loss = 0
        decoder_attention_loss = 0
        encoder_embedding_loss = 0
        decoder_embedding_loss = 0
        encoder_layer_loss = 0
        decoder_layer_loss = 0
        prediction_layer_loss = 0
        
        # Calculate losses in encoder layers
        for i in range(num_encoder_layers_student):
            encoder_attention_loss += MSELoss(
                # attentions before softmax should be used(should change later if possible)
                outputs_student.encoder_attentions[i], 
                outputs_teacher.encoder_attentions[i*encoder_layer_ratio+1]
            )

        # Calculate losses in decoder layers
        for i in range(num_decoder_layers_student):
            decoder_attention_loss += MSELoss(
                # attentions before softmax should be used(should change later if possible)
                outputs_student.decoder_attentions[i],
                outputs_teacher.decoder_attentions[i*decoder_layer_ratio+1]
            )
        
        # Calculate encoder embedding loss
        encoder_embedding_loss = MSELoss(
            outputs_student.encoder_hidden_states[0],
            outputs_teacher.encoder_hidden_states[0],
        )

        # Calculate decoder embedding loss
        decoder_embedding_loss = MSELoss(
            outputs_student.decoder_hidden_states[0],
            outputs_teacher.decoder_hidden_states[0],
        )
        
        # Calculate encoder layer loss
        for i in range(1, num_encoder_layers_student+1):
            encoder_layer_loss += MSELoss(
                outputs_student.encoder_hidden_states[i], 
                outputs_teacher.encoder_hidden_states[i*encoder_layer_ratio]
            )

        # Calculate decoder layer loss
        for i in range(1, num_decoder_layers_student+1):
            decoder_layer_loss += MSELoss(
                outputs_student.decoder_hidden_states[i], 
                outputs_teacher.decoder_hidden_states[i*decoder_layer_ratio]
            )


        # Calculate prediction with KD
        loss_fct = nn.KLDivLoss(reduction="batchmean")
        loss_kd = self.args.temperature ** 2 * loss_fct(
            F.log_softmax(outputs_student.logits/ self.args.temperature, dim=-1),
            F.softmax(outputs_teacher.logits/ self.args.temperature, dim=-1)
            )
        prediction_layer_loss = self.args.alpha * outputs_student.loss + (1. - self.args.alpha) * loss_kd
        
        loss = (encoder_attention_loss +
                decoder_attention_loss +
                encoder_embedding_loss +
                decoder_embedding_loss +
                encoder_layer_loss +
                decoder_layer_loss +
                prediction_layer_loss
        )
        
        if labels is not None:
            loss += self.label_smoothed_nll_loss(outputs_student, labels,
                                            epsilon=0.1 if self.label_smoother else 0)
    
        if self.args.use_rdrop:
            loss += self.args.reg_alpha * self.compute_kl_loss(outputs_student, pad_mask=pad_mask)

        return (loss, outputs_student) if return_outputs else loss
    
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