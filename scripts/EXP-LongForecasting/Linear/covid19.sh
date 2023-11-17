
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
# seq_len=336
seq_len=512
# model_name=DLinear
model_name=Autoformer

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path covid-19.csv \
  --model_id Covid19_$seq_len'_'96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 24 \
  --des 'Exp' \
  --use_gpu True \
  --train_epochs 100 --patience 10\
  --individual --itr 1 --batch_size 16 --learning_rate 0.0001 >>  logs/LongForecasting/$model_name'-'individual'_'covid19_$seq_len'_'96.log # 

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path covid-19.csv \
  --model_id Covid19_$seq_len'_'192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 24 \
  --des 'Exp' \
  --use_gpu True \
  --individual --train_epochs 100 --patience 10\
  --itr 1 --batch_size 16  --learning_rate 0.001 >> logs/LongForecasting/$model_name'-'individual'_'covid19_$seq_len'_'192.log  

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path covid-19.csv \
  --model_id Covid19_$seq_len'_'336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 24 \
  --des 'Exp' \
  --use_gpu True \
  --individual --train_epochs 100 --patience 10\
  --itr 1 --batch_size 16 --learning_rate 0.001  >> logs/LongForecasting/$model_name'-'individual'_'covid19_$seq_len'_'336.log  

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path covid-19.csv \
  --model_id Covid19_$seq_len'_'720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 24 \
  --des 'Exp' \
  --use_gpu True \
  --individual --train_epochs 100 --patience 10\
  --itr 1 --batch_size 16 --learning_rate 0.001 >> logs/LongForecasting/$model_name'-'individual'_'covid19_$seq_len'_'720.log  
