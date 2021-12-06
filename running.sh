# python train.py \
# --do_train \
# --dataset_name paper,news,magazine,law \
# --output_dir model/kobart \
# --num_train_epochs 2 \
# --learning_rate 3e-05 \
# --max_source_length 1024 \
# --max_target_length 128 \
# --metric_for_best_model rougeLsum \
# --es_patience 5 \
# --relative_eval_steps 20 \
# --wandb_unique_tag kobart_ep2_lr3e05_srclen1024_tgtlen128

# python train.py \
# --do_eval \
# --model_name_or_path model/kobart \
# --dataset_name news \
# --output_dir evaluation/kobart_beam5_ml0 \
# --num_beams 5 \
# --wandb_unique_tag kobart_ep2_lr3e05_srclen1024_tgtlen128_eval_b5_ml0


python train.py \
--do_eval \
--model_name_or_path hyunwoongko/kobart \
--dataset_name paper,news,magazine,law \
--output_dir evaluation/kobart_default \
--wandb_unique_tag kobart_default_evaluation