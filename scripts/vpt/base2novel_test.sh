#!/bin/bash

# custom config
DATA=/mnt/hdd/zhurui/data
TRAINER=VPT

DATASET=$1
CFG=$2  # config file 例如 vit_b16
SHOTS=$3 
SEED=$4 # random seed

# eg. bash scripts/vpt/base2novel_test.sh caltech101 vit_b16 16 1
DIR=output/${DATASET}/${TRAINER}/base2novel_test/${CFG}/shots_${SHOTS}/seed_${SEED}
CUDA_VISIBLE_DEVICES=0 python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir output/${DATASET}/${TRAINER}/base2novel_train/${CFG}/shots_${SHOTS}/seed_${SEED} \
    --eval-only \
    TRAINER.MODAL base2novel \
    DATASET.SUBSAMPLE_CLASSES new
    # --load-epoch 50 \ # 可以选择对应轮数的模型，此处直接使用最佳的模型进行测试，故无需该参数

