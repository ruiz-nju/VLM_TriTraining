#!/bin/bash

# custom config
DATA=/data0/zhur/data
TRAINER=CoOp

DATASET=$1
CFG=$2  # config file 例如 vit_b16
SHOTS=$3 
SEED=$4 # random seed
DEVICE=$5


DIR=output/reproduce/${DATASET}/${TRAINER}/base2novel_test/${CFG}/shots_${SHOTS}/seed_${SEED}
CUDA_VISIBLE_DEVICES=$DEVICE python train.py \
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

