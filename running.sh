## 변경사항: 법률 데이터 미포함, 데이터셋 비율 50%, 정리를 위한 wandb project 변경
## 시도해볼 부분: epoch 수정해보기
## 변경 필요한 arguments: output_dir
python train.py \
--do_train \
--output_dir model/kobart_bV1 \
--dataset_name paper,news,magazine \
--num_train_epochs 3 \
--learning_rate 3e-05 \
--max_source_length 1024 \
--max_target_length 128 \
--metric_for_best_model rougeLsum \
--relative_eval_steps 10 \
--es_patience 3 \
--load_best_model_at_end True \
--relative_sample_ratio 0.5 \
--project_name Kobart_bV1.0 \
--wandb_unique_tag kobartV1_ep2_lr3e05_len1024_R50

# ## 시도해볼 부분: num_beams
# ## 변경 필요한 model_name_or_path: output_dir
# python train.py \
# --do_eval \
# --model_name_or_path model/kobart_bV1 \
# --dataset_name paper,news,magazine \
# --output_dir evaluation/kobart_eval \
# --num_beams 3 \
# --relative_sample_ratio 1 \
# --project_name Kobart_bV1.0 \
# --wandb_unique_tag Eval_kobartV1_ep2_lr3e05_len1024_R50