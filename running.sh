## 변경사항: 법률 데이터 미포함, 데이터셋 비율 50%, 정리를 위한 wandb project 변경

## Pretraining using Infilling
## 학습 파라미터 : epoch, weight decay, learning rate, warmup steps

python pretrain.py \
--do_train \
--is_pretrain \
--output_dir model/longformerbart_bV1 \
--overwrite_output_dir True \
--num_train_epochs 10 \
--weight_decay 1e-5 \
--warmup_steps 20000 \
--learning_rate 2e-04 \
--max_source_length 2048 \
--max_target_length 2048 \
--project_name baseV1.0_Kobart \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 4 \
--wandb_unique_tag longformerBart_pretraining \
--hidden_size 128 \
--encoder_layer_size 3 \
--decoder_layer_size 3 \
--attention_head_size 4 \
--attention_window_size 32 \
--num_samples 10 \
--dropout 0.5





# 1. h_dim 128/256 => 논문 => 128 / 256
# 2. layer depth 3/1 6/3 => 논문 => 3/3
# +a window_size, head = 32 / 64 => (4)
# 3. dropout 70/50/30 => 선택 => 50/70
# 4. weight decay 1e-5 => fix
# 5. warmup => 
# 5. teacher forcing -> lr 형태로 100 -> 0 => (구현 필요) -> 해야죠 => fine_tuning => 내일
# 6. LR scheduler => noam => (구현) -> 끝
# 7. LR : +-1e-4



