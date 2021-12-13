## 변경사항: 법률 데이터 미포함, 데이터셋 비율 50%, 정리를 위한 wandb project 변경

## Train
## 시도해볼 부분: epoch 수정해보기
## 변경 필요한 arguments: output_dir
python train.py \
--do_train \
--output_dir model/klue_roberta \
--overwrite_output_dir \
--config_name klue/roberta-base --model_name_or_path klue/roberta-base \
--dataset_name paper,news,magazine \
--num_train_epochs 2 \
--learning_rate 3e-05 \
--max_source_length 512 \
--max_target_length 128 \
--metric_for_best_model rougeLsum \
--relative_eval_steps 10 \
--preprocessing_num_workers 1 \
--es_patience 3 \
--load_best_model_at_end True \
--relative_sample_ratio 0.5 \
--project_name baseV1.0_Kobart \
--wandb_unique_tag Roberta_ep2_lr3e05_len1024_R50_ConsineWithWarmup

## Eval
# ## 시도해볼 부분: num_beams
# ## 변경 필요한 model_name_or_path: output_dir
# python train.py \
# --do_eval \
# --model_name_or_path model/baseV1.0_Kobart \
# --dataset_name paper,news,magazine \
# --output_dir evaluation/kobart_eval \
# --num_beams 3 \
# --relative_sample_ratio 1 \
# --project_name baseV1.0_Kobart \
# --wandb_unique_tag Eval_kobartV1_ep2_lr3e05_len1024_R50

# ## Predict
# python predict.py \
# --model_name_or_path baseV1.0_Kobart \
# --num_beams 3
