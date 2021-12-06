rm -rf ./model/*

python train.py \
--do_train \
--dataset_name paper,news,magazine,law \
--output_dir results/kobart \
--num_train_epochs 2 \
--learning_rate 3e-05 \
--max_source_length 1024 \
--max_target_length 128 \
--metric_for_best_model rougeLsum \
--es_patience 3 \
--preprocessing_num_workers 1 \
--relative_eval_steps 10 \
--max_eval_samples 10000 \
--wandb_unique_tag kobart_ep2_lr3e05_srclen1024_tgtlen128_preprocessing

# python train.py \
# --do_eval \
# --model_name_or_path model/kobart \
# --dataset_name paper,news,magazine,law \
# --output_dir evaluation/kobart_beam3_ml0 \
# --num_beams 3 \
# --wandb_unique_tag kobart_ep2_lr3e05_srclen1024_tgtlen128_eval_b3_ml0