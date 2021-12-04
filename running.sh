python train.py \
--do_train \
--output_dir model/kobart \
--num_train_epochs 2 \
--learning_rate 3e-05 \
--dataset_name news \
--max_source_length 1024 \
--max_target_length 128 \
--wandb_unique_tag kobart_ep2_lr3e05_srclen1024_tgtlen128

# --dataset_name paper,news,magazine,law \
