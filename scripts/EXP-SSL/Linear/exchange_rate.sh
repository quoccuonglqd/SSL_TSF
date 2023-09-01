
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/SSL" ]; then
    mkdir ./logs/SSL
fi
seq_len=96
# model_name=DLinear
model_name=Ours
ssl_model_name=BYOL

python -u run_ssl.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'96 \
  --model $model_name \
  --ssl_model_id $ssl_model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 8 \
  --use_gpu True \
  --des 'Exp' \
  --individual \
  --itr 1 --batch_size 8 --learning_rate 0.0005 >logs/SSL/$model_name'_'Exchange_$seq_len'_'96.log 

python -u run_ssl.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'192 \
  --model $model_name \
  --ssl_model_id $ssl_model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 8 \
  --use_gpu True \
  --des 'Exp' \
  --individual \
  --itr 1 --batch_size 8 --learning_rate 0.0005 >logs/SSL/$model_name'_'Exchange_$seq_len'_'192.log 

python -u run_ssl.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'336 \
  --model $model_name \
  --ssl_model_id $ssl_model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 8 \
  --use_gpu True \
  --des 'Exp' \
  --individual \
  --itr 1 --batch_size 32  --learning_rate 0.0005 >logs/SSL/$model_name'_'Exchange_$seq_len'_'336.log 

python -u run_ssl.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'720 \
  --model $model_name \
  --ssl_model_id $ssl_model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 8 \
  --use_gpu True \
  --des 'Exp' \
  --individual \
  --itr 1 --batch_size 32 --learning_rate 0.0005 >logs/SSL/$model_name'_'Exchange_$seq_len'_'720.log
