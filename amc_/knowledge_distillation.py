import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Seq2SeqTrainingArguments
from transformers import Trainer

class DistillationTrainingArguments(Seq2SeqTrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature

class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
    
    def compute_loss(self, model, inputs, return_outputs=False):
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
        
        print('loss:', round(loss.item(), 4), 'loss_ce:', round(loss_cross_entropy.item(), 4), 'loss_kd:', round(loss_kd.item(), 4))
        # Return weighted student loss
        return (loss, outputs_student) if return_outputs else loss

class TinyTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
    
    def compute_loss(self, model, inputs):
        pass