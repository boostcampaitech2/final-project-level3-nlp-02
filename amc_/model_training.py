#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import datasets
import numpy as np
import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from knowledge_distillation import DistillationTrainingArguments, DistillationTrainer
from huggingface_hub import notebook_login
from dotenv import load_dotenv
load_dotenv(verbose=True)
import torch
import wandb
import sys
sys.path.append('/opt/ml/final-project-level3-nlp-02')
from rouge import compute

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
wandb.init(
    entity="final_project",
)


# In[28]:


seed = 42

train_size = 5000
eval_size = 500
check_point = "gogamza/kobart-summarization"
max_input_length = 512
max_target_length = 30

batch_size = 8
num_train_epochs = 2
learning_rate=5.6e-5
weight_decay = 0.01
logging_steps = 100
model_name = check_point.split("/")[-1]

is_distillation = True
if is_distillation:
    student_check_point = "encoder_decoder_pruned_last_3"
    teacher_check_point = "kobart-summarization-finetuned-paper-sample-size-1000/checkpoint-1000"
    alpha=0.5
    temperature = 2.0


# # Loading Dataset

# In[29]:


api_token = os.getenv('HF_DATASET_API_TOKEN')
dataset = datasets.load_dataset('metamong1/summarization_paper', use_auth_token=api_token)


# In[30]:


training_dataset = dataset['train'].shuffle(seed=seed).select(range(train_size))


# # Prepare Training

# In[31]:


if is_distillation:
    tokenizer = AutoTokenizer.from_pretrained(student_check_point)
else:
    tokenizer = AutoTokenizer.from_pretrained(check_point)


# In[32]:


def preprocess_function(examples):
    model_inputs = tokenizer(
        examples['text'], max_length=max_input_length, truncation = True, #padding=True
    )

    # Set up the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples['title'], max_length=max_target_length, truncation=True, #padding=True
        )
    
    model_inputs['labels'] = labels['input_ids']

    return model_inputs


# In[33]:


tokenized_train_dataset = training_dataset.map(preprocess_function, batched=True)


# In[34]:


tokenized_eval_dataset = dataset['validation'].select(range(500)).map(preprocess_function, batched=True)


# # Training

# In[35]:


if is_distillation:
    student_model = AutoModelForSeq2SeqLM.from_pretrained(student_check_point).to(device)
    teacher_model = AutoModelForSeq2SeqLM.from_pretrained(teacher_check_point).to(device)
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(check_point).to(device)


# In[36]:


if is_distillation:
    args =  DistillationTrainingArguments(
        output_dir=f'{student_check_point}-distilled-{train_size}',
        evaluation_strategy = 'steps',
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay = weight_decay,
        save_total_limit=2,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        logging_steps=logging_steps,
        alpha=alpha,
        temperature=temperature,
        report_to='wandb'
    )
else:
    args = Seq2SeqTrainingArguments(
        output_dir=f'{model_name}-finetuned-{train_size}',
        evaluation_strategy='steps', #'epoch',
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay = weight_decay,
        save_total_limit=2,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        logging_steps=logging_steps,
        report_to='wandb'
        # push_to_hub=True,
    )


# In[37]:


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute ROUGE scores
    result = compute(
        predictions=decoded_preds, references=decoded_labels, tokenizer=tokenizer
    )

    # Extract the median scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}


# In[38]:


if is_distillation:
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=student_model)
else:
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


# In[39]:


if is_distillation:
    trainer = DistillationTrainer(
        model=student_model,
        args=args,
        teacher_model = teacher_model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
else:
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics, 
    )


# In[40]:


trainer.train()


# In[ ]:




