fit_epoch=$1
CUDA_VISIBLE_DEVICES=0 python tritraining_main.py \
    --fit_epoch ${fit_epoch} 