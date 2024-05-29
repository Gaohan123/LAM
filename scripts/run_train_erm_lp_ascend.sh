export ASCEND_RT_VISIBLE_DEVICES=0
unset RANK_TABLE_FILE
export DEVICE_ID=0
export RANK_SIZE=1

msrun --bind_core=True --worker_num=1 --local_worker_num=1 --log_dir=./device_erm_lp python train.py \
      --config_path=config/vit_patch32_imagenet9_erm_lp.yml