config=$1

# training
python3 -m torch.distributed.launch --master_port=12098 --nproc_per_node=8 train.py --config ${config}

sleep 50s

# evaluate
python3 -m torch.distributed.launch --master_port=12097 --nproc_per_node=8 evaluate.py --config ${config} --start_epoch 0 --end_epoch 47