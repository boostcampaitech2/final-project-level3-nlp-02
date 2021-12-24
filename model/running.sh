########## pretraining #################
python pretrain.py \
--do_train \
--is_pretrain \
--output_dir model/longformerbart_pretrain_V1_trial3 \
--num_train_epochs 10 \
--logging_steps 2000 \
--save_strategy epoch \
--evaluation_strategy no \
--max_source_length 2048 \
--max_target_length 2048 \
--project_name longformerbart \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 4 \
--wandb_unique_tag longformerBart_pretraining_V1 \
--hidden_size 128 \
--encoder_layer_size 3 \
--decoder_layer_size 3 \
--attention_head_size 4 \
--attention_window_size 32 \
--dropout 0.5 \
--learning_rate 0.11 \
--warmup_steps 10000 \
--weight_decay 1e-2 \
--adam_beta1  0.9 \
--adam_beta2  0.999 \
--adam_epsilon 1e-06 \
--num_samples 10 \
--is_noam

########## training #################
python train.py \
--model_name_or_path metamong1/bigbird-tapt-ep3 \
--use_model bigbart \
--do_train \
--output_dir checkpoint/kobigbirdbart_full_tapt_ep3_bs16_pre_RD \
--overwrite_output_dir \
--num_train_epochs 3 \
--learning_rate 1e-4 \
--max_source_length 4096 \
--max_target_length 128 \
--metric_for_best_model rougeLsum \
--es_patience 3 \
--load_best_model_at_end True \
--project_name kobigbirdbart \
--wandb_unique_tag kobigbirdbart_full_tapt_ep3_bs16_pre_RD \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 16 \
--gradient_accumulation_steps 16 \
--use_preprocessing True \
--label_smoothing_factor 0.1 \
--use_rdrop True \
--evaluation_strategy epoch \
--save_strategy epoch \
--use_doc_type_ids True

# ########## Knowledge Distillation #################
python train.py \
--model_name_or_path tmp/bigbart_full_tapt_ep3_bs16_pre_RD_half_warmpstep_pruned_3 \
--do_train \
--use_model bigbart \
--output_dir checkpoint/bigbart_full_tapt_ep3_bs16_pre_RD_half_warmpstep_pruned_3_distil \
--overwrite_output_dir \
--num_train_epochs 3 \
--learning_rate 1e-4 \
--max_source_length 4096 \
--max_target_length 128 \
--overwrite_output_dir \
--metric_for_best_model rougeLsum \
--es_patience 3 \
--load_best_model_at_end \
--project_name optimization \
<<<<<<< HEAD
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16 \
--per_device_eval_batch_size 16 \
--evaluation_strategy epoch \
--save_strategy epoch \
--distillation_type tiny \
--is_warmup_half \
--use_rdrop \
--label_smoothing_factor 0.1 \
--teacher_check_point metamong1/bigbart_full_tapt_ep3_bs16_pre_RD_half_warmpstep \
--wandb_unique_tag bigbart_full_tapt_ep3_bs16_pre_RD_half_warmpstep_pruned_3_distil \
--use_preprocessing


########## Test #################
python test.py \
--model_name_or_path metamong1/bigbart_tapt_ep3_bs16_pre_noam \
--output_dir result/test \
--overwrite_output_dir \
--use_model bigbart_tapt \
--use_doc_type_ids \
--use_preprocessing \
--per_device_eval_batch_size 32 \
--wandb_unique_tag tapt_ep3_bs16_pre_noam_vaild \
--max_source_length 4096 \
--max_target_length 128 \
--is_valid

########## Prediction #################
python predict.py \
--model_name_or_path checkpoint/kobigbirdbart_base_ep3_bs8_pre_noam \
--num_beams 3 \
--use_model bigbart \
--use_preprocessing
=======
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 8 \
--per_device_eval_batch_size 32 \
--is_noam \
--evaluation_strategy epoch \
--save_strategy epoch \
--distillation_type tiny \
--warmup_steps 1000 \
--teacher_check_point metamong1/bigbart_tapt_ep3_bs16_pre_noam \
--wandb_unique_tag bigbart_tapt_ep3_bs16_pre_noam_tiny_full \
--use_preprocessing
>>>>>>> c2c4580d73adc096c42a6d6267cea3ac3755fde2
