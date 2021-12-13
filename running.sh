## 변경사항: 법률 데이터 미포함, 데이터셋 비율 50%, 정리를 위한 wandb project 변경

## Pretraining using Infilling
## 학습 파라미터 : epoch, weight decay, learning rate, warmup steps

python pretrain.py \
--do_train \
--is_pretrain \
--output_dir model/longformerbart_bV1 \
--overwrite_output_dir True \
--num_train_epochs 10 \
--weight_decay 1e-2 \
--warmup_steps 20000 \
--learning_rate 2e-04 \
--max_source_length 2048 \
--max_target_length 2048 \
# --num_samples 10 \
--project_name baseV1.0_Kobart \
--per_device_train_batch_size 1 \
--warmup_ratio 0.5 \
--wandb_unique_tag longformerBart_t1_attn128_2048_2048_2e04_warmup50%_pretraining \
--is_part
