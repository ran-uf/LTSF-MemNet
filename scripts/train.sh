if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
seq_len=96
model_name=MemNet
for pred_len in 96 192 336 729
do
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id Electricity_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 321 \
  --des 'Exp' \
  --itr 1 --batch_size 256  --learning_rate 0.005 --individual >logs/$model_name'_I_'electricity_$seq_len'_'$pred_len.log 
