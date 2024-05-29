export ASCEND_RT_VISIBLE_DEVICES=0
unset RANK_TABLE_FILE
export DEVICE_ID=0
export RANK_SIZE=1

msrun --bind_core=True --worker_num=1 --local_worker_num=1 --log_dir=./device_eval python eval.py \
      --config_path=config/vit_eval.yml
