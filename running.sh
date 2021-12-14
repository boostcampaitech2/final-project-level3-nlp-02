## 변경사항: 법률 데이터 미포함, 데이터셋 비율 50%, 정리를 위한 wandb project 변경

## Train
## 시도해볼 부분: epoch 수정해보기
## 변경 필요한 arguments: output_dir
python train.py \
--do_train \
--output_dir model/baseV1.0_Kobart \
--num_train_epochs 3 \
--learning_rate 3e-05 \
--max_source_length 1024 \
--max_target_length 128 \
--metric_for_best_model rougeLsum \
--relative_eval_steps 10 \
--es_patience 3 \
--load_best_model_at_end True \
--project_name baseV1.0_Kobart \
--wandb_unique_tag kobartV1_ep3_lr3e05_len1024_R50_rdrop1.0 \
--use_rdrop True \
--label_smoothing_factor 0.1

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
# --model_name_or_path model/baseV1.0_Kobart_ep2_1210 \
# --num_beams 3
