
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting_Finetune" ]; then
    mkdir ./logs/LongForecasting_Finetune
fi
seq_len=96
# model_name=DLinear
model_name=Ours

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 8 \
  --use_gpu True \
  --des 'Exp' \
  --individual \
  --finetune \
  --pretrained_path checkpoints/BYOL-Individual_Exchange_96_96_Ours_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0 \
  --itr 1 --batch_size 8 --learning_rate 0.0005 >logs/LongForecasting_Finetune/$model_name'_'Exchange_$seq_len'_'96.log 

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 8 \
  --use_gpu True \
  --des 'Exp' \
  --individual \
  --finetune \
  --pretrained_path checkpoints/BYOL-Individual_Exchange_96_192_Ours_custom_ftM_sl96_ll48_pl192_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0 \
  --itr 1 --batch_size 8 --learning_rate 0.0005 >logs/LongForecasting_Finetune/$model_name'_'Exchange_$seq_len'_'192.log 

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 8 \
  --use_gpu True \
  --des 'Exp' \
  --individual \
  --finetune \
  --pretrained_path checkpoints/BYOL-Individual_Exchange_96_336_Ours_custom_ftM_sl96_ll48_pl336_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0 \
  --itr 1 --batch_size 32  --learning_rate 0.0005 >logs/LongForecasting_Finetune/$model_name'_'Exchange_$seq_len'_'336.log 

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 8 \
  --use_gpu True \
  --des 'Exp' \
  --individual \
  --finetune \
  --pretrained_path checkpoints/BYOL-Individual_Exchange_96_720_Ours_custom_ftM_sl96_ll48_pl720_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0 \
  --itr 1 --batch_size 32 --learning_rate 0.0005 >logs/LongForecasting_Finetune/$model_name'_'Exchange_$seq_len'_'720.log
