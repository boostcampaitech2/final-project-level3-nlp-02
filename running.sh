## 법률 데이터 미포함, 데이터셋 비율 50%, 정리를 위한 wandb project 변경
python train.py \
--do_train \
--dataset_name paper,news,magazine \
--output_dir model/es_test \
--num_train_epochs 2 \
--learning_rate 3e-05 \
--max_source_length 1024 \
--max_target_length 128 \
--metric_for_best_model rougeLsum \
--relative_eval_steps 10 \
--es_patience 2 \
--load_best_model_at_end True \
--relative_sample_ratio 0.001 \
--project_name reproducibility_test \
--wandb_unique_tag cache_test_model_init2


# python train.py \
# --do_train \
# --dataset_name paper,news,magazine \
# --output_dir model/kobart-v1 \
# --num_train_epochs 2 \
# --learning_rate 3e-05 \
# --max_source_length 1024 \
# --max_target_length 128 \
# --metric_for_best_model rougeLsum \
# --es_patience 3 \
# --relative_eval_steps 10 \
# --relative_sample_ratio 0.5 \
# --project_name kobart_1207 \
# --wandb_unique_tag kobartV1_ep2_lr3e05_srclen1024_R50

# python train.py \
# --do_eval \
# --model_name_or_path model/kobart \
# --dataset_name paper,news,magazine,law \
# --output_dir evaluation/kobart_beam3_ml0 \
# --num_beams 3 \
# --wandb_unique_tag kobart_ep2_lr3e05_srclen1024_tgtlen128_eval_b3_ml0