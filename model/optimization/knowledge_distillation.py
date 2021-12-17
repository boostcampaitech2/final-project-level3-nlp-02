import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from nn_pruning.sparse_trainer import SparseTrainer

class DistillationTrainingArguments(Seq2SeqTrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, use_original=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature
        self.use_original = use_original

class DistillationTrainer(Seq2SeqTrainer):
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
        
        # Return weighted student loss
        return (loss, outputs_student) if return_outputs else loss

class TinyTrainer(Seq2SeqTrainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
    
    def compute_loss(self, model, inputs, return_outputs=False):
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

        # Calculate prediction layer loss
        if self.args.use_original:
            prediction_layer_loss = -F.softmax(outputs_teacher.logits, dim=-1) * F.log_softmax(outputs_student.logits, dim=-1)
            prediction_layer_loss = prediction_layer_loss.mean()
        elif not self.args.use_original:
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
        
        return (loss, outputs_student) if return_outputs else loss

class PruningTrainer(SparseTrainer, Seq2SeqTrainer):
    def __init__(self, sparse_args, *args, **kwargs):
        Seq2SeqTrainer.__init__(self, *args, **kwargs)
        SparseTrainer.__init__(self, sparse_args)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        We override the default loss in SparseTrainer because it throws an 
        error when run without distillation
        """
        outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        self.metrics["ce_loss"] += float(loss)
        self.loss_counter += 1
        return (loss, outputs) if return_outputs else loss
    
    